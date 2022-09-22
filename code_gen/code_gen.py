beginning =r"""
  BEGIN_ASM()
"""

ending =r"""

    VZEROUPPER()

  END_ASM
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(A),
   [b] "m"(B),
   [c] "m"(C),
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", reglist "memory"
  )
"""

def initReg(numReg, sizeReg):
    regdict = dict()
    regdict[128] = 'XMM'
    regdict[256] = 'YMM'
    regdict[512] = 'ZMM'
    outcode =r""""""
    tmpcode =r""""""
    setregzero = "VXORPD(regname( idreg), regname( idreg), regname( idreg))"
    for i in range(numReg):
       tmpcode =           setregzero.replace("idreg",str(i))
       outcode = outcode + tmpcode.replace("regname",regdict[sizeReg]) + "\n"

    return(outcode)

def setupRegs():
    outcode=r""""""
    outcode = outcode + "MOV(RSI, VAR(k))" + "\n"
    outcode = outcode + "MOV(RAX, VAR(a))" + "\n"
    outcode = outcode + "MOV(RBX, VAR(b))" + "\n"
    outcode = outcode + "MOV(RCX, VAR(c))" + "\n"
    return(outcode)

beginning =r"""
  BEGIN_ASM()
"""

ending =r"""

    VZEROUPPER()

  END_ASM
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(A),
   [b] "m"(B),
   [c] "m"(C),
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", reglist "memory"
  )
"""


def initReg(numReg, sizeReg):
    regdict = dict()
    regdict[128] = 'XMM'
    regdict[256] = 'YMM'
    regdict[512] = 'ZMM'
    outcode =r""""""
    tmpcode =r""""""
    setregzero = "VXORPD(regname( idreg), regname( idreg), regname( idreg))"
    for i in range(numReg):
       tmpcode =           setregzero.replace("idreg",str(i))
       outcode = outcode + tmpcode.replace("regname",regdict[sizeReg]) + "\n"

    return(outcode)


def setupRegs():
    outcode=r""""""
    outcode = outcode + "MOV(RSI, VAR(k))" + "\n"
    outcode = outcode + "MOV(RAX, VAR(a))" + "\n"
    outcode = outcode + "MOV(RBX, VAR(b))" + "\n"
    outcode = outcode + "MOV(RCX, VAR(c))" + "\n"
    return(outcode)

def mainLoop(numReg, sizeReg, unrollFactor):
    regdict = dict()
    regdict[128] = "XMM"
    regdict[256] = "YMM"
    regdict[512] = "ZMM"
    regdictlc = dict()
    regdictlc[128] = "xmm"
    regdictlc[256] = "ymm"
    regdictlc[512] = "zmm"
    nelemInReg = sizeReg//64
    prefetch  = "PREFETCH(0, MEM(regname, idfac2*8))"
    loadmtor  = "VMOVUPD(REG( idreg1), MEM(regname, idfac2*8))"
    loadrtom  = "VMOVUPD(MEM(regname, idfac2*8), REG( idreg1))"
    add       = "VADDPD(REG( idreg1), REG( idreg2), MEM(regname, idfac2*8))"
    broadcast = "VBROADCASTSD(REG( idreg1), MEM(regname, idfac2*8))"
    fma       = "VFMADD231PD(REG( idreg1), REG( idreg2), REG( idreg3))"
    lea       = "LEA(regname, MEM(regname, jmpfac*8))"
    outcode =r""""""
    finalcode =r""""""
    for ik in range(unrollFactor):
        outcode =r""""""
        # Main loop
        NR = (numReg - 4)//2
        MR = 2 * (sizeReg//64)
        tmpload = loadmtor.replace("REG",regdict[sizeReg])
        tmpload = tmpload.replace("regname","RAX")
        tmpload = tmpload.replace("szereg",str(sizeReg))
        tmpload = tmpload.replace("idreg1",str(0))
        tmpload = tmpload.replace("idfac2",str(0))
        outcode = "\n" + "\t" + outcode + tmpload + "\n"
        tmpload = loadmtor.replace("REG",regdict[sizeReg])
        tmpload = tmpload.replace("regname","RAX")
        tmpload = tmpload.replace("szereg",str(sizeReg))
        tmpload = tmpload.replace("idreg1",str(1))
        tmpload = tmpload.replace("idfac2",str(MR//2))
        outcode = outcode + "\t" + tmpload + "\n" + "\n"
        regid = 3
        facb = 0
        for nrid in range(NR//2):
            tmpmov = broadcast.replace("REG",regdict[sizeReg])
            tmpmov = tmpmov.replace("regname","RBX")
            tmpmov = tmpmov.replace("idreg1",str(2))
            tmpmov = tmpmov.replace("idfac2",str(facb))
            outcode = outcode + "\t" + tmpmov + "\n"

            tmpmov = broadcast.replace("REG",regdict[sizeReg])
            tmpmov = tmpmov.replace("regname","RBX")
            tmpmov = tmpmov.replace("idreg1",str(3))
            tmpmov = tmpmov.replace("idfac2",str(facb + 1))
            outcode = outcode + "\t" + tmpmov + "\n"
            facb = facb + 2

            regid += 1
            tmpfma = fma.replace("REG",regdict[sizeReg])
            tmpfma = tmpfma.replace("idreg1",str(regid))
            tmpfma = tmpfma.replace("idreg2",str(0))
            tmpfma = tmpfma.replace("idreg3",str(2))
            outcode = outcode + "\t" + tmpfma + "\n"

            regid += 1
            tmpfma = fma.replace("REG",regdict[sizeReg])
            tmpfma = tmpfma.replace("idreg1",str(regid))
            tmpfma = tmpfma.replace("idreg2",str(1))
            tmpfma = tmpfma.replace("idreg3",str(2))
            outcode = outcode + "\t" + tmpfma + "\n"

            regid += 1
            tmpfma = fma.replace("REG",regdict[sizeReg])
            tmpfma = tmpfma.replace("idreg1",str(regid))
            tmpfma = tmpfma.replace("idreg2",str(0))
            tmpfma = tmpfma.replace("idreg3",str(3))
            outcode = outcode + "\t" + tmpfma + "\n"

            regid += 1
            tmpfma = fma.replace("REG",regdict[sizeReg])
            tmpfma = tmpfma.replace("idreg1",str(regid))
            tmpfma = tmpfma.replace("idreg2",str(1))
            tmpfma = tmpfma.replace("idreg3",str(3))
            outcode = outcode + "\t" + tmpfma + "\n"

        tmplea = "\n\t" + lea.replace("regname","RAX")
        tmplea = tmplea.replace("jmpfac",str(MR))
        outcode = outcode + "\t" + tmplea + "\n"

        tmplea = lea.replace("regname","RBX")
        tmplea = tmplea.replace("jmpfac",str(NR))
        outcode = outcode + "\t" + tmplea + "\n\n"

        finalcode = finalcode + outcode

    outcode =r""""""
    idxreg = 4
    for inr in range(NR):
        tmppref = prefetch.replace("regname","RCX")
        tmppref = tmppref.replace("idfac2",str(192))
        outcode = outcode + tmppref + "\n"

        tmpadd = add.replace("regname","RCX")
        tmpadd = tmpadd.replace("REG",regdict[sizeReg])
        tmpadd = tmpadd.replace("idreg1",str(1))
        tmpadd = tmpadd.replace("idreg2",str(idxreg))
        tmpadd = tmpadd.replace("idfac2",str(0))
        outcode = outcode + tmpadd + "\n"
        idxreg = idxreg + 1

        tmpload = loadrtom.replace("REG",regdict[sizeReg])
        tmpload = tmpload.replace("regname","RCX")
        tmpload = tmpload.replace("szereg",str(sizeReg))
        tmpload = tmpload.replace("idreg1",str(1))
        tmpload = tmpload.replace("idfac2",str(0))
        outcode = outcode + tmpload + "\n"

        #tmppref = prefetch.replace("regname","RCX")
        #tmppref = tmppref.replace("idfac2",str(192))
        #outcode = outcode + tmppref + "\n"

        tmpadd = add.replace("regname","RCX")
        tmpadd = tmpadd.replace("REG",regdict[sizeReg])
        tmpadd = tmpadd.replace("idreg1",str(1))
        tmpadd = tmpadd.replace("idreg2",str(idxreg))
        tmpadd = tmpadd.replace("idfac2",str(MR//2))
        outcode = outcode + tmpadd + "\n"
        idxreg = idxreg + 1

        tmpload = loadrtom.replace("REG",regdict[sizeReg])
        tmpload = tmpload.replace("regname","RCX")
        tmpload = tmpload.replace("szereg",str(sizeReg))
        tmpload = tmpload.replace("idreg1",str(1))
        tmpload = tmpload.replace("idfac2",str(MR//2))
        outcode = outcode + tmpload + "\n"

        tmplea = "\n\t" + lea.replace("regname","RCX")
        tmplea = tmplea.replace("jmpfac",str(MR))
        outcode = outcode + "\t" + tmplea + "\n"


    header =r"""
    TEST(RSI, RSI)
    JE(K_LOOP)""" + "\n\t" + "LABEL(LOOP1)\n\n"

    tailer =r"""
    DEC(RSI)
    JNE(LOOP1)
    """ + "\nLABEL(K_LOOP)\n\n"

    finalcode = header + finalcode + tailer + outcode

    # Ending
    reglist = r""""""
    for i in range(numReg):
        reglist = reglist + "\"" + regdictlc[sizeReg] + str(i) + "\"" + ", "

    return(beginning + initReg(numReg,sizeReg) + setupRegs() + finalcode + ending.replace("reglist",reglist))

print(mainLoop(32,512,2))
