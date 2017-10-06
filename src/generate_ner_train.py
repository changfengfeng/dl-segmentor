# -*- coding: utf-8 -*-
import sys
import os
import word2vec as w2v
import traceback

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
        chari.append("0")
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

def processNERToken(token_block, sentence, out, pos_vob):
    found_ner = False
    nn = len(token_block)
    while nn > 0 and token_block[nn - 1] != '/':
        nn = nn -1
    pos = token_block[nn:]
    if len(pos) > 2:
        pos = pos[:2]
    if (not pos[0:1].isalpha()) or pos[0:1].isupper():
        sentence.markWrong = True
        return False

    if token_block.startswith('['):
        token_len = len(token_block)
        while token_len > 0 and token_block[token_len - 1] != ']':
            token_len = token_len -1
        token_block = token_block[1:token_len - 1]
        # deal with evyer word in the block
        ss = token_block.split(' ')
        for i in range(len(ss)):
            token = ss[i]
            slen = len(token)
            while slen > 0 and token[slen - 1] != '/':
                slen = slen - 1
            token = token[:slen - 1].strip()
            if pos in pos_vob:
                found_ner = True
                if i == 0:
                    sentence.addToken(Token(token, pos_vob[pos]))
                elif i + 1 == len(ss):
                    sentence.addToken(Token(token, pos_vob[pos] + 2))
                else:
                    sentence.addToken(Token(token, pos_vob[pos] + 1))
            else:
                sentence.addToken(Token(token, 0))
    else:
        token = token_block[:nn-1].strip()
        if pos in pos_vob:
            sentence.addToken(Token(token, pos_vob[pos] + 3))
            found_ner = True
        else:
            sentence.addToken(Token(token, 0))

    return found_ner

def processLine(line, out, word_vob, char_vob, pos_vob):
  line = line.strip()
  nn = len(line)
  seeLeftB = False
  start = 0
  sentence = Sentence()
  found_ner = False
  try:
    for i in range(nn):
      if line[i] == ' ':
        if not seeLeftB:
          token = line[start:i]
          if processNERToken(token, sentence, out, pos_vob):
              found_ner = True
          start = i + 1
      elif line[i] == '[':
        seeLeftB = True
      elif line[i] == ']':
        seeLeftB = False
    if start < nn:
      token = line[start:]
      if processNERToken(token, sentence, out, pos_vob):
          found_ner = True
    if found_ner and (not sentence.markWrong) and len(sentence.tokens) > 0:
      sentence.generate_train_line(out, word_vob, char_vob)
  except Exception as e:
    print(e)
    traceback.print_exc()

def main(argc, argv):
  global totalLine
  global longLine
  global totalChars
  if argc < 5:
    print("Usage:%s <word_vob> <char_vob> <dir> <output>" %
          (argv[0]))
    sys.exit(1)
  wvobPath = argv[1]
  cvobPath = argv[2]
  rootDir = argv[3]
  print(wvobPath, cvobPath, rootDir)

  w2v_model = w2v.load(wvobPath)
  c2v_model = w2v.load(cvobPath)
  word_vob = WordVectWrapper(w2v_model.vocab)
  char_vob = WordVectWrapper(c2v_model.vocab)

  posVob = {"nt": 1, "ns": 5, "nr": 9 }
  out = open(argv[4], "w")

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
