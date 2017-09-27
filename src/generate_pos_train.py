# -*- coding: utf-8 -*-
import sys
import os
import word2vec as w2v

print(sys.version)

totalLine = 0
longLine = 0

MAX_LEN = 50
totalChars = 0

class WordVectWrapper:
    """ Wrapper on word2vec, provide GetWordIndex method
    """
    def __init__(self, vob):
        """
        Args:
            vob: the vob of the word2vec
        """
        self.word_map = {w:i for i, w in enumerate(vob)}
        print("wrapper ", len(self.word_map))

    def GetWordIndex(self, w):
        if w in self.word_map:
            return self.word_map[w]
        else:
            # return the unkown word idx
            return self.word_map['<UNK>']


class Sentence:
  def __init__(self):
    self.tokens = []
    self.markWrong = False

  def addToken(self, token):
    self.tokens.append(token)

  def generate_train_line(self, out, word_vob, char_vob):
    nl = len(self.tokens)
    if nl < 3:
      return
    wordi = []
    chari = []
    labeli = []
    if nl > MAX_LEN:
      nl = MAX_LEN
    for ti in range(nl):
      t = self.tokens[ti]
      idx = word_vob.GetWordIndex(t.token)
      wordi.append(str(idx))
      labeli.append(str(t.posTag))
      nc = len(t.chars)
      if nc > 5:
        lc = t.chars[nc - 1]
        t.chars[4] = lc
        nc = 5
      for i in range(nc):
        idx = char_vob.GetWordIndex(t.chars[i])
        chari.append(str(idx))
      for i in range(nc, 5):
        chari.append("0")
    for i in range(nl, MAX_LEN):
      wordi.append("0")
      labeli.append("0")
      for ii in range(5):
        chari.append(str(ii))
    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    line += " ".join(labeli)
    out.write("%s\n" % (line))

class Token:
  def __init__(self, token, posTag):
    self.token = token
    ustr = token
    self.chars = []
    for u in ustr:
      self.chars.append(u)
    self.posTag = posTag

def processToken(token, sentence, out, end, word_vob, char_vob, pos_vob):
  global totalLine
  global longLine
  global totalChars
  global MAX_LEN
  nn = len(token)
  while nn > 0 and token[nn - 1] != '/':
    nn = nn - 1
  pos = token[nn:]
  if (not pos[0:1].isalpha()) or pos[0:1].isupper():
    sentence.markWrong = True
    return False
  if len(pos) > 2:
    pos = pos[:2]
  if pos not in pos_vob:
    print("mark wrong for:[%s]" % (pos))
    sentence.markWrong = True
    return False
  token = token[:nn - 1].strip()
  sentence.addToken(Token(token, pos_vob[pos]))
  return True

def processLine(line, out, word_vob, char_vob, pos_vob):
  line = line.strip()
  nn = len(line)
  seeLeftB = False
  start = 0
  sentence = Sentence()
  try:
    for i in range(nn):
      if line[i] == ' ':
        if not seeLeftB:
          token = line[start:i]
          if token.startswith('['):
            tokenLen = len(token)
            while tokenLen > 0 and token[tokenLen - 1] != ']':
              tokenLen = tokenLen - 1
            token = token[1:tokenLen - 1]
            ss = token.split(' ')
            for s in ss:
              processToken(s, sentence, out, False, word_vob, char_vob,
                           pos_vob)
          else:
            processToken(token, sentence, out, False, word_vob, char_vob,
                         pos_vob)
          start = i + 1
      elif line[i] == '[':
        seeLeftB = True
      elif line[i] == ']':
        seeLeftB = False
    if start < nn:
      token = line[start:]
      if token.startswith('['):
        tokenLen = len(token)
        while tokenLen > 0 and token[tokenLen - 1] != ']':
          tokenLen = tokenLen - 1
        token = token[1:tokenLen - 1]
        ss = token.split(' ')
        ns = len(ss)
        for i in range(ns - 1):
          processToken(ss[i], sentence, out, False, word_vob, char_vob,
                       pos_vob)
        processToken(ss[-1], sentence, out, True, word_vob, char_vob, pos_vob)
      else:
        processToken(token, sentence, out, True, word_vob, char_vob, pos_vob)
    if (not sentence.markWrong) and len(sentence.tokens) > 0:
      # OKay,generating training line
      sentence.generate_train_line(out, word_vob, char_vob)
  except Exception as e:
    print(e)

def loadPosVob(path, vob):
  fp = open(path, "r")
  for line in fp.readlines():
    line = line.strip()
    if not line:
      continue
    ss = line.split("\t")
    vob[ss[0]] = int(ss[1])
  pass

def main(argc, argv):
  global totalLine
  global longLine
  global totalChars
  if argc < 6:
    print("Usage:%s <word_vob> <char_vob> <pos_vob>  <dir> <output>" %
          (argv[0]))
    sys.exit(1)
  wvobPath = argv[1]
  cvobPath = argv[2]
  pvobPath = argv[3]
  rootDir = argv[4]
  print(wvobPath, cvobPath, pvobPath)

  w2v_model = w2v.load(wvobPath)
  c2v_model = w2v.load(cvobPath)
  word_vob = WordVectWrapper(w2v_model.vocab)
  char_vob = WordVectWrapper(c2v_model.vocab)

  posVob = {}
  loadPosVob(pvobPath, posVob)
  out = open(argv[5], "w")

  print("word2vec: ", w2v_model.vectors.shape)
  print("char2vec: ", c2v_model.vectors.shape)

  for dirName, subdirList, fileList in os.walk(rootDir):
    curDir = dirName
    for file in fileList:
      if file.endswith(".txt"):
        curFile = os.path.join(curDir, file)
        print("processing:%s" % (curFile))
        fp = open(curFile, "r")
        for line in fp.readlines():
          line = line.strip()
          processLine(line, out, word_vob, char_vob, posVob)
        fp.close()
  out.close()
  print("total:%d, long lines:%d, chars:%d" %
        (totalLine, longLine, totalChars))

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
