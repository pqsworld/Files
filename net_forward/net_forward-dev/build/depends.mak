

ifeq ($(ONEDIR),1)                                 ##### TARGET_BASE
 ifeq ($(CLANG),1)
      TARGET_BASE=output/$(TARGET_NAME)-clang
 else
      TARGET_BASE=output/$(TARGET_NAME)
 endif
else
 ifeq ($(ARMCC),1)
      TARGET_BASE=output/qsee#rvds
 else
  ifeq ($(CLANG),1)
      TARGET_BASE=output/qsee4/lib32#clang
      ifeq ($(SNAPDRAGON),1)
        QSEE4_SECURITY_COMPILE := 1
      endif
  else
   ifeq ($(AARCH64),1)
	  # ifeq ($(LINUX),1)#linux64
    #     TARGET_BASE=output/linux-lib/lib64
		# endif

    ifeq ($(QSEE),1)
      TARGET_BASE=output/qsee4/lib64#qsee64
      QSEE4_SECURITY_COMPILE := 1

    else ifeq ($(WINDOWS),1)
      TARGET_BASE=output/windows/lib64
    else
      ifeq ($(GCCVER),7.3)
        TARGET_BASE=output/tee-VII/lib64#aarch64 7.3
      else ifeq ($(GCCVER),8.3)
        TARGET_BASE=output/tee-VIII/lib64#aarch64 8.3
      else ifeq ($(GCCVER),6.3)
        TARGET_BASE=output/tee-VI/lib64#aarch64 6.3
      else ifeq ($(GCCVER),9.2)
        TARGET_BASE=output/tee-IX/lib64#aarch64 9.2
      else ifeq ($(GCCVER),clang64_13)
        TARGET_BASE=output/clang13/lib64
      else
        TARGET_BASE=output/tee-lib/lib64#aarch64
      endif


    endif  # QSEE=1
   else
      ifeq ($(WINDOWS),1)
        TARGET_BASE=output/windows/lib32
      else ifeq ($(GCCVER),8.3)
        TARGET_BASE=output/tee-VIII/lib32#gcc8.3
      else ifeq ($(GCCVER),6.3)
        TARGET_BASE=output/tee-VI/lib32#gcc6.3
      else ifeq ($(GCCVER),9.2)
        TARGET_BASE=output/tee-IX/lib32#gcc6.3
      else
        TARGET_BASE=output/tee-lib/lib32#gcc
      endif

   endif # AARCH64=1
  endif # CLANG=1
 endif # ARMCC=1
endif # ONEDIR
TARGET_CON=_
COMMON_INCLUDES= -I. -I./include -I./include/cust
LFLAGS=
INCLUDES=
HISI_SECURITY_COMPILE := 1
ifeq ($(ARMCC),1) 																	##### ARMCC
    CC=armcc
    AR=armar
    CFLAGS1= -DNEON=0 --apcs=/ropi/rwpi/softfp --lower_ropi --no_unaligned_access --cpu=QSP.no_neon.no_vfp --c99 --thumb -O2 -Otime -W -DNDEBUG --split-sections
    CFLAGS2= --cpu=Cortex-A7 --fpu=vfpv4 --fpmode=fast --arm --c99 --thumb --apcs=/ropi/rwpi --lower_ropi --lower_rwpi --protect_stack --apcs /noswst/interwork --littleend --force_new_nothrow --dwarf2 -Otime -O2 --enum_is_int --interface_enums_are_32_bit -DNEON=0
    CFLAGS3= -DNEON=0 --apcs=/ropi/rwpi/softfp --lower_ropi --no_unaligned_access --cpu=QSP.no_neon.no_vfp --c99 --thumb -O2 -Otime -W -DNDEBUG --split-sections --interface_enums_are_32_bit --enum_is_int
    TARGET_TZ=qsee
    TYPE1=$(TARGET_TZ)1
    TYPE2=$(TARGET_TZ)2
    TYPE3=$(TARGET_TZ)3
    LFLAGS=-r
    XFLAGS=-x
   ifeq ($(ONEDIR),1)
    TARGET1=$(TARGET_BASE)$(TARGET_CON)$(TYPE1)
    TARGET2=$(TARGET_BASE)$(TARGET_CON)$(TYPE2)
    TARGET3=$(TARGET_BASE)$(TARGET_CON)$(TYPE3)
   else
    TARGET1=$(TARGET_BASE)/$(TYPE1)/$(TARGET_NAME)
    TARGET2=$(TARGET_BASE)/$(TYPE2)/$(TARGET_NAME)
    TARGET3=$(TARGET_BASE)/$(TYPE3)/$(TARGET_NAME)
   endif
    COMMON_CFLAGS+= -DSECURE_QSEE -DCUST_H=\"custmazaanaaa.h\"
    COMMON_INCLUDES+= -I./include/qsee3
else

   ifeq ($(GCCVER),)
    GCCVER=4.8
   endif
   ifneq ($(CLANG),1)
    ifeq ($(GCCVER),$(filter $(GCCVER),8.3 9.2))
      CFLAGS_GCC1= -Wno-implicit-fallthrough -Wno-stringop-overflow
      ifeq ($(GCCVER),9.2)
       CFLAGS_GCC3= -fno-builtin-calloc -w -Wno-address-of-packed-member
       CFLAGS_GCC1+= $(CFLAGS_GCC3)
      endif
    else
     CFLAGS_GCC1= -DSECURE_TZ -fstack-check
     #CFLAGS_GCC2= $(CFLAGS_GCC1) -fenum-int-equiv
     CFLAGS_GCC2= $(CFLAGS_GCC1)
     CFLAGS_GCC3= -DSECURE_TZ
    endif
   else
    ifeq ($(SNAPDRAGON),1)
     CFLAGS_GCC1= -DSNAPDRAGON_LLVM -w -D__ARM_PCS_VFP -DARM_CLANG -fpic
     CFLAGS_GCC2= $(CFLAGS_GCC1)
     CFLAGS_GCC3= -DSNAPDRAGON_LLVM -w -D__ARM_PCS_VFP -DARM_CLANG
    else
     #CFLAGS_GCC1= -DARM_CLANG
     #CFLAGS_GCC2= $(CFLAGS_GCC1)
     #CFLAGS_GCC3= $(CFLAGS_GCC1)
     CFLAGS_GCC1= -fstack-check -D__ARM_PCS_VFP -DARM_CLANG -fpic -DSECURE_TZ
     #CFLAGS_GCC2= $(CFLAGS_GCC1) -fenum-int-equiv
     CFLAGS_GCC2= $(CFLAGS_GCC1)
     CFLAGS_GCC3= -D__ARM_PCS_VFP -DARM_CLANG -DSECURE_TZ
     ALGO_SUBDIR=qsee4/
    endif
   endif

   ifneq ($(AARCH64),1)
    ifeq ($(GCCVER),8.3)
     CFLAGS_GCC1= -std=c99 -mthumb-interwork -fstack-usage -O2 -fdata-sections -ffunction-sections -fno-short-enums -nostdlib -Wall -Wextra -DPLAT=ARMV7_A_STD -DARM_ARCH=ARMv7 -D__ARMv7__ -D__ARMV7_A__ -D__ARMV7_A_STD__ -Werror -U__INT32_TYPE__ -U__UINT32_TYPE__  -D__INT32_TYPE__="int" -fstack-protector-strong -fpie -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mthumb  -Wno-builtin-macro-redefined -Wincompatible-pointer-types -Wstringop-truncation -Werror=format-truncation  -Wno-error -Wno-format-truncation
     CFLAGS1= $(CFLAGS_GCC1)
     CFLAGS2= $(CFLAGS_GCC1)
     CFLAGS3= $(CFLAGS_GCC1)
     CFLAGS4= $(CFLAGS_GCC1)
     CFLAGS5= $(CFLAGS_GCC1)
     CFLAGS6= $(CFLAGS_GCC1)
    else ifeq ($(GCCVER),9.2)
     CFLAGS1= $(CFLAGS_GCC1) -O2 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fshort-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mthumb
     CFLAGS2= $(CFLAGS_GCC1) -O2 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fshort-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfloat-abi=softfp -mthumb -DNEON=0
     CFLAGS3= $(CFLAGS_GCC1) -O3 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fno-short-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mthumb
     CFLAGS4= $(CFLAGS_GCC1) -O3 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fno-short-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfloat-abi=softfp -mthumb -DNEON=0
     CFLAGS5= $(CFLAGS_GCC1) -O3 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fno-short-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfloat-abi=softfp -mthumb -DNEON=0
     CFLAGS6= $(CFLAGS_GCC1) -O3 -std=c99 -mthumb-interwork -fdata-sections -ffunction-sections -fno-short-enums -nostdlib -Wextra -fstack-protector-strong -fpie -march=armv7-a -mfloat-abi=softfp -mthumb -DNEON=0
    else
     CFLAGS1= $(CFLAGS_GCC1) -O2 -mfpu=neon -fno-omit-frame-pointer -mfloat-abi=softfp -mthumb-interwork -fno-strict-aliasing -fno-short-wchar -fshort-enums -Wcast-align
     CFLAGS2= $(CFLAGS_GCC1) -O2 -fno-omit-frame-pointer -mfloat-abi=softfp -mthumb-interwork -fno-strict-aliasing -fno-short-wchar -fshort-enums -Wcast-align -DNEON=0
     CFLAGS3= $(CFLAGS_GCC2) -O3 -mfpu=neon -fno-omit-frame-pointer -mfloat-abi=softfp -mthumb-interwork -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align
     CFLAGS4= $(CFLAGS_GCC2) -O2 -mcpu=cortex-a15 -mfpu=neon -fno-omit-frame-pointer -mfloat-abi=softfp -mthumb-interwork -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align
     #CFLAGS4= $(CFLAGS_GCC2) -O2 -fno-omit-frame-pointer -mfloat-abi=softfp -mthumb-interwork -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align -DNEON=0
     ifeq ($(SENSORHUB),arc)
      CFLAGS5= $(CFLAGS_GCC3) -g -O0 -Ml -Hnosdata -Hnocopyr -ffunction-sections -Wno-extra-tokens -Wall -arcv2hs -core4 -Xdual_issue -Xcode_density -HL -Xatomic -Xll64 -Xunaligned -Xdiv_rem=radix4 -Xswap -Xbitscan -Xmpy_option=qmpyh -Xshift_assist -Xbarrel_shifter  -Xdsp3  -Xdsp_complex -Xdsp_divsqrt=radix4 -Xdsp_itu -Xdsp_accshift=full -Xdsp_wide -Xfpus_div -Xfpu_mac -Xstack_check -Hld_cycles=1 -Hccm -Xlpb_size=64 -DDYN_RESERVED_PAGE_CNT=0 -D__MW__
     else
      CFLAGS5= $(CFLAGS_GCC3) -O2 -fpic -mno-unaligned-access -march=armv5te -mfpu=vfpv3-d16 -fdata-sections -ffunction-sections -fomit-frame-pointer -mfloat-abi=softfp -fno-strict-aliasing -DNEON=0
     endif # SENSORHUB = 1
     CFLAGS6= -std=gnu99 -fdiagnostics-show-option -Wall -Wstrict-aliasing=2 -mcpu=cortex-a15 -Os -g3 -fpie -mthumb -mthumb-interwork -fno-short-enums -fno-common -mno-unaligned-access -mfloat-abi=softfp  -mfpu=neon-vfpv4 -D__ARM_PCS_VFP
     ifeq ($(HISI_SECURITY_COMPILE),1)
      CFLAGS8= $(CFLAGS1) -fstack-protector-all
      CFLAGS9= $(CFLAGS2) -fstack-protector-all
     endif # HISI_SECURITY_COMPILE 1
     ifeq ($(QSEE4_SECURITY_COMPILE),1)
      CFLAGS8= $(CFLAGS_GCC1) -O2 -mfpu=neon -fno-omit-frame-pointer -mfloat-abi=softfp -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align -fstack-protector-all
     endif # QSEE4_SECURITY_COMPILE 1
    endif
   else # 64bit
     CFLAGS_GCC1= -fstack-check -D__ARM_PCS_VFP -DARM_CLANG -fpic -DAARCH64_SUPPORT -Wno-int-to-pointer-cast
     ifeq ($(GCCVER),9.2)
      CFLAGS_GCC1+= -w -Wno-address-of-packed-member
     endif
     CFLAGS_GCC2= $(CFLAGS_GCC1)
     CFLAGS_GCC3= $(CFLAGS_GCC1)
     CFLAGS2= $(CFLAGS_GCC1) -O2 -fno-omit-frame-pointer -fno-strict-aliasing -fno-short-wchar -fshort-enums -Wcast-align
     CFLAGS4= $(CFLAGS_GCC2) -O2 -fno-omit-frame-pointer -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align
     CFLAGS6= $(CFLAGS_GCC3) -O2 -fpic -fdata-sections -ffunction-sections -fomit-frame-pointer -fno-strict-aliasing
     ifeq ($(HISI_SECURITY_COMPILE),1)
      CFLAGS9= $(CFLAGS2) -fstack-protector-all
     endif # HISI_SECURITY_COMPILE 1
     ifeq ($(QSEE4_SECURITY_COMPILE),1)
      CFLAGS9= $(CFLAGS2) -fstack-protector-all
     endif # QSEE4_SECURITY_COMPILE 1
     ifeq ($(GCCVER),clang64_13)
      CFLAGS2= -pipe -Wall -Wextra -Wdate-time -Wfloat-equal -Wshadow -Wformat=2 -Wstack-protector -fsigned-char -fno-common -fPIC -funwind-tables -munaligned-access -fmax-type-align=1 -flto -fno-exceptions -Wno-int-to-pointer-cast -O2 -fno-omit-frame-pointer -fno-strict-aliasing -fno-short-wchar -fshort-enums -Wcast-align  -DSIL_DUMP_TA_LOG -DSECURE_QSEE -DSIZE_MAX=1024*1024 -DCUST_H=\"custmazaanaaa.h\" -march=armv8.5-a -DSIL_STRING --target=aarch64-linux-gnu -DARM_CLANG -DGNU_LLVM -DAARCH64_SUPPORT -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined -fno-builtin-calloc -Wno-format-nonliteral -Wno-unused-function  -DSIL_TEST_ENABLE
      CFLAGS4= -pipe -Wall -Wextra -Wdate-time -Wfloat-equal -Wshadow -Wformat=2 -Wstack-protector -fsigned-char -fno-common -fPIC -funwind-tables -munaligned-access -fmax-type-align=1 -flto -fno-exceptions -Wno-int-to-pointer-cast -O2 -fno-omit-frame-pointer -fno-strict-aliasing -fno-short-wchar -fno-short-enums -Wcast-align  -DSIL_DUMP_TA_LOG -DSECURE_QSEE -DSIZE_MAX=1024*1024 -DCUST_H=\"custmazaanaaa.h\" -march=armv8.5-a -DSIL_STRING --target=aarch64-linux-gnu -DARM_CLANG -DGNU_LLVM -DAARCH64_SUPPORT -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined -fno-builtin-calloc -Wno-format-nonliteral -Wno-unused-function  -DSIL_TEST_ENABLE
      CFLAGS6= -pipe -Wall -Wextra -Wdate-time -Wfloat-equal -Wshadow -Wformat=2 -Wstack-protector -fsigned-char -fno-common -fPIC -funwind-tables -munaligned-access -fmax-type-align=1 -flto -fno-exceptions -Wno-int-to-pointer-cast -O2 -fpic -fdata-sections -ffunction-sections -fomit-frame-pointer -fno-strict-aliasing  -DSIL_DUMP_TA_LOG -DSECURE_QSEE -DSIZE_MAX=1024*1024 -DCUST_H=\"custmazaanaaa.h\" -march=armv8.5-a -DSIL_STRING --target=aarch64-linux-gnu -DARM_CLANG -DGNU_LLVM -DAARCH64_SUPPORT -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined -fno-builtin-calloc -Wno-format-nonliteral -Wno-unused-function  -DSIL_TEST_ENABLE
      CFLAGS7= -mcpu=cortex-a53 -mno-unaligned-access -Wno-tautological-constant-out-of-range-compare -mharden-sls=all -Os -O2 -fPIC -ffunction-sections -mbranch-protection=pac-ret+b-key+bti  -DSIL_DUMP_TA_LOG -DSECURE_QSEE -DSIZE_MAX=1024*1024 -DCUST_H=\"custmazaanaaa.h\" -march=armv8.5-a -DSIL_STRING --target=aarch64-linux-gnu -DARM_CLANG -DGNU_LLVM -DAARCH64_SUPPORT -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined -fno-builtin-calloc -Wno-format-nonliteral -Wno-unused-function  -DSIL_TEST_ENABLE
     endif #clang64_13
   endif
    TARGET_TZ=x
    TYPE1=neon$(TARGET_CON)$(TARGET_TZ)1
    TYPE2=$(TARGET_TZ)1
    TYPE3=ei_neon$(TARGET_CON)$(TARGET_TZ)1
    TYPE4=ei$(TARGET_CON)$(TARGET_TZ)1
    TYPE5=neon$(TARGET_CON)$(TARGET_TZ)2
    TYPE6=$(TARGET_TZ)2
    TYPE7=$(TARGET_TZ)3
    TYPE8=sec$(TARGET_CON)$(TYPE1)
    TYPE9=sec$(TARGET_CON)$(TYPE2)
   ifeq ($(ONEDIR),1)
    TARGET1=$(TARGET_BASE)$(TARGET_CON)$(TYPE1)
    TARGET2=$(TARGET_BASE)$(TARGET_CON)$(TYPE2)
    TARGET3=$(TARGET_BASE)$(TARGET_CON)$(TYPE3)
    TARGET4=$(TARGET_BASE)$(TARGET_CON)$(TYPE4)
    TARGET5=$(TARGET_BASE)$(TARGET_CON)$(TYPE5)
    TARGET6=$(TARGET_BASE)$(TARGET_CON)$(TYPE6)
    TARGET7=$(TARGET_BASE)$(TARGET_CON)$(TYPE7)
    TARGET8=$(TARGET_BASE)$(TARGET_CON)$(TYPE8)
    TARGET9=$(TARGET_BASE)$(TARGET_CON)$(TYPE9)
   else
    TARGET1=$(TARGET_BASE)/$(TYPE1)/$(TARGET_NAME)
    TARGET2=$(TARGET_BASE)/$(TYPE2)/$(TARGET_NAME)
    TARGET3=$(TARGET_BASE)/$(TYPE3)/$(TARGET_NAME)
    TARGET4=$(TARGET_BASE)/$(TYPE4)/$(TARGET_NAME)
    TARGET5=$(TARGET_BASE)/$(TYPE5)/$(TARGET_NAME)
    TARGET6=$(TARGET_BASE)/$(TYPE6)/$(TARGET_NAME)
    TARGET7=$(TARGET_BASE)/$(TYPE7)/$(TARGET_NAME)
    TARGET8=$(TARGET_BASE)/$(TYPE8)/$(TARGET_NAME)
    TARGET9=$(TARGET_BASE)/$(TYPE9)/$(TARGET_NAME)
   endif
   ifeq ($(TEE),1)
    COMMON_CFLAGS+= -DSECURE_TEE -DCUST_H=\"custmazaanaaa.h\"
    ifeq ($(GCCVER),$(filter $(GCCVER),8.3 9.2))  #t-SDK 500
      COMMON_CFLAGS+= -DSECURE_TEE5
    endif
   else
    ifeq ($(GP),1)
      COMMON_CFLAGS+= -DSECURE_GP -DCUST_H=\"custmazaanaaa.h\"
      ifeq ($(GCCVER),$(filter $(GCCVER),8.3 9.2))  #t-SDK 500
        COMMON_CFLAGS+= -DSECURE_GP5
      endif
    else
      COMMON_CFLAGS+= -DSECURE_QSEE -DSIZE_MAX=1024*1024 -DCUST_H=\"custmazaanaaa.h\"
    endif
   endif
   ifeq ($(WINDOWS),1)                                                                   ##### win32库
    CFLAGS10 = -O2 -DNEON=0 -msse2 -mfpmath=sse
    TARGET10=$(TARGET_BASE)/$(TARGET_NAME)
    LFLAGS=-r
   endif
   ifeq ($(LINUX),1)                                                                     ##### linux32库
	  COMMON_CFLAGS :=
		COMMON_INCLUDES :=
    CFLAGS10 = -O2 -DNEON=0
		TARGET_BASE = output/linux-lib/lib64
    TARGET10=$(TARGET_BASE)/$(TARGET_NAME)
    LFLAGS=-r
   endif
 ifeq ($(CLANG),1)
  ifeq ($(SNAPDRAGON),1)
    TOOLCHAIN=armv7-linux-gnueabi/libc
    CROSS_GCC_PATH=/apps/snapdragon/10.0.9
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/
    CC=$(CROSS_COMPILE)clang
    AR=$(CROSS_COMPILE)llvm-ar
    LD=$(CROSS_COMPILE)llvm-link
    COMMON_CFLAGS+= -target armv8-linux-gnueabi
    COMMON_INCLUDES+= -I$(CROSS_GCC_PATH)/$(TOOLCHAIN)/include
    CFLAG_ASM= -mcpu=cortex-a9
  else
    TOOLCHAIN=arm-linux-gnueabihf
    CROSS_GCC_PATH=/apps/linaro/gcc-linaro-arm-linux-gnueabihf-4.9-2014.07_linux
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/4.9.1
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
  endif
    LFLAGS=-r
    XFLAGS=-x
    COMMON_INCLUDES+= -I./include/qsee4
 else
  ifeq ($(AARCH64),1)

    ifeq ($(WINDOWS),1)
      CROSS_COMPILE=x86_64-w64-mingw32-
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
      LD=$(CROSS_COMPILE)ld
    else ifeq ($(GCCVER),7.3)
      TOOLCHAIN=aarch64-linux-gnu
      CROSS_GCC_PATH=/apps/linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/7.3.1
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
      COMMON_CFLAGS+=-fPIC
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
    else ifeq ($(GCCVER),6.3)
      TOOLCHAIN=aarch64-secureos-gnueabi
      CROSS_GCC_PATH=/apps/teegris_sdk.4.1.0/toolchains/aarch64-secureos-gnueabi-gcc_6_3-linux-x86
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/6.3
      COMMON_INCLUDES += -I/apps/teegris_sdk.4.1.0/platforms/TEEGRIS-4.1/swd/arch-arm64/usr/include -I/apps/teegris_sdk.4.1.0/toolchains/aarch64-secureos-gnueabi-gcc_6_3-linux-x86/lib/gcc/aarch64-secureos-gnueabi/6.3.1/include
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
    else ifeq ($(GCCVER),8.3)
      TOOLCHAIN=aarch64-elf
      CROSS_GCC_PATH=/apps/gcc-arm-8.3-2019.03-x86_64-aarch64-elf
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/8.3.0
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
      LD=$(CROSS_COMPILE)ld
      COMMON_CFLAGS+= -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined
    else ifeq ($(GCCVER),9.2)
      TOOLCHAIN=aarch64-none-elf
      CROSS_GCC_PATH=/apps/fp-toolchain/linaro/gcc-arm-9.2-2019.12-x86_64-aarch64-none-elf
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/$(GCCVER).1
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
      LD=$(CROSS_COMPILE)ld
      COMMON_CFLAGS+= -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined
    else ifeq ($(GCCVER),clang64_13)
      CLANGVER=13.0.0
      CROSS_GCC_PATH=/apps/fp-toolchain/llvm/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/lib/
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/clang/$(CLANGVER)
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/
      CC=$(CROSS_COMPILE)clang
      AR=$(CROSS_COMPILE)llvm-ar
      LD=$(CROSS_COMPILE)llvm-link
      COMMON_CFLAGS+= -march=armv8.5-a -DSIL_STRING
      COMMON_CFLAGS+= --target=aarch64-linux-gnu -DARM_CLANG=$(word 1, $(subst .,  , $(CLANGVER)))
      COMMON_CFLAGS+= -DGNU_LLVM -DAARCH64_SUPPORT -U__INT64_TYPE__ -U__UINT64_TYPE__ -D__INT64_TYPE__="long long" -D__intptr_t_defined -D__uintptr_t_defined -fno-builtin-calloc -Wno-format-nonliteral -Wno-unused-function
      COMMON_INCLUDES+= -I$(CROSS_GCC_PATH_LGCC)/include
    else
      TOOLCHAIN=aarch64-none-elf
      CROSS_GCC_PATH=/apps/linaro/gcc-linaro-aarch64-none-elf-4.9-2014.07_linux
      CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
      CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/4.9.1
      CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
      CC=$(CROSS_COMPILE)gcc
      AR=$(CROSS_COMPILE)ar
    endif
   ifeq ($(QSEE),1)
    COMMON_INCLUDES+= -I./include/qsee4
   endif
  else
   ifeq ($(SENSORHUB),arc)
    TOOLCHAIN=ac
    CROSS_GCC_PATH=/apps/ARC/MetaWare/arc
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/bin/lib32
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/av2hs/le
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/
    CC=$(CROSS_COMPILE)ccac
    AR=$(CROSS_COMPILE)arac
    LD=$(CROSS_COMPILE)ldac
    LFLAGS=-r
    XFLAGS=-x
   else ifeq ($(WINDOWS),1)
    CROSS_COMPILE=i686-w64-mingw32-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
    LD=$(CROSS_COMPILE)ld
   else ifeq ($(GCCVER),6.3)
    TOOLCHAIN=arm-secureos-gnueabi
    CROSS_GCC_PATH=/apps/teegris_sdk.4.1.0/toolchains/arm-secureos-gnueabi-gcc_6_3-linux-x86
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/6.3.1
    COMMON_INCLUDES += -I/apps/teegris_sdk.4.1.0/platforms/TEEGRIS-4.1/swd/arch-arm/usr/include -I/apps/teegris_sdk.4.1.0/toolchains/arm-secureos-gnueabi-gcc_6_3-linux-x86/lib/gcc/arm-secureos-gnueabi/6.3.1/include
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
    LD=$(CROSS_COMPILE)ld
   else ifeq ($(GCCVER),8.3)
    TOOLCHAIN=arm-eabi
    CROSS_GCC_PATH=/apps/gcc-arm-8.3-2019.03-x86_64-arm-eabi
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/8.3.0
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
    LD=$(CROSS_COMPILE)ld
    LFLAGS=-r
    XFLAGS=-x
    COMMON_CFLAGS+= -U__INT32_TYPE__ -U__UINT32_TYPE__ -D__INT32_TYPE__="int"
   else ifeq ($(GCCVER),9.2)
    TOOLCHAIN=arm-none-eabi
    CROSS_GCC_PATH=/apps/fp-toolchain/linaro/gcc-arm-9.2-2019.12-x86_64-arm-none-eabi
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/$(GCCVER).1
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
    LD=$(CROSS_COMPILE)ld
    LFLAGS=-r
    XFLAGS=-x
    COMMON_CFLAGS+= -U__INT32_TYPE__ -U__UINT32_TYPE__ -D__INT32_TYPE__="int"
   else ifeq ($(GCCVER),4.8)
    CFLAGS1 += -DNOT_PLD=1
    CFLAGS3 += -DNOT_PLD=1
    CFLAGS5 += -DNOT_PLD=1
    CFLAGS8 += -DNOT_PLD=1
    TOOLCHAIN=arm-none-eabi
    CROSS_GCC_PATH=/apps/gcc-arm-none-eabi-4_8-2014q2
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/4.8.4
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
   else
    TOOLCHAIN=arm-linux-androideabi
    CROSS_GCC_PATH=/apps/mt6757-n/prebuilts/gcc/linux-x86/arm/arm-linux-androideabi-4.9
    CROSS_GCC_PATH_LIB=$(CROSS_GCC_PATH)/$(TOOLCHAIN)/lib
    CROSS_GCC_PATH_LGCC=$(CROSS_GCC_PATH)/lib/gcc/$(TOOLCHAIN)/4.9
    CROSS_COMPILE=$(CROSS_GCC_PATH)/bin/$(TOOLCHAIN)-
    CC=$(CROSS_COMPILE)gcc
    AR=$(CROSS_COMPILE)ar
   endif  #GCCVER=4.8
  endif # AARCH64=1

    LFLAGS=-r
    XFLAGS=-x
 endif
endif

ifeq ($(LINUX),1)
  CROSS_COMPILE=
  CC=$(CROSS_COMPILE)gcc
  AR=$(CROSS_COMPILE)ar
  LD=$(CROSS_COMPILE)ld
endif

CFLAGS_depend= -fstack-check -D__ARM_PCS_VFP -DARM_CLANG -fpic -DSECURE_TZ
CC_depend=/apps/linaro/gcc-linaro-aarch64-none-elf-4.9-2014.07_linux/bin/aarch64-none-elf-gcc

ifeq ($(EXT),1)
    CFLAGS = $(CFLAGS1)
    TARGET = $(TARGET1)
    TYPE   = $(TYPE1)
else ifeq ($(EXT),2)
    CFLAGS = $(CFLAGS2)
    TARGET = $(TARGET2)
    TYPE   = $(TYPE2)
else ifeq ($(EXT),3)
    CFLAGS = $(CFLAGS3)
    TARGET = $(TARGET3)
    TYPE   = $(TYPE3)
else ifeq ($(EXT),4)
    CFLAGS = $(CFLAGS4)
    TARGET = $(TARGET4)
    TYPE   = $(TYPE4)
else ifeq ($(EXT),5)
    CFLAGS = $(CFLAGS5)
    TARGET = $(TARGET5)
    TYPE   = $(TYPE5)
else ifeq ($(EXT),6)
    CFLAGS = $(CFLAGS6)
    TARGET = $(TARGET6)
    TYPE   = $(TYPE6)
else ifeq ($(EXT),7)
    CFLAGS = $(CFLAGS7)
    TARGET = $(TARGET7)
    TYPE   = $(TYPE7)
else ifeq ($(EXT),8)
    CFLAGS = $(CFLAGS8) $(CFLAG_ASM)
    TARGET = $(TARGET8)
    TYPE   = $(TYPE8)
else ifeq ($(EXT),9)
    CFLAGS = $(CFLAGS9)
    TARGET = $(TARGET9)
    TYPE   = $(TYPE9)
else ifeq ($(EXT),10)
    CFLAGS = $(CFLAGS10)
    TARGET = $(TARGET10)
else
    CFLAGS = $(CFLAGS1)
    TARGET = $(TARGET1)
    TYPE   = $(TYPE1)
endif
CFLAGS += -D$(PROJECT) -D$(CUST)
