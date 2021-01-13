import os
import sys
import argparse
import subprocess

# default parameters and templates

MOVI_COMPILE_COMMAND_TEMPLATE = '-mcpu=ma2x8x -O3 -ffunction-sections -Wall -Wextra -Werror -S {pathToKernel} -I ./include -o {outFolder}/kernel.asmgen'
MOVI_ASM_COMMAND_TEMPLATE = '-cv:ma2x8x -noSPrefixing -nowarn:20 {outFolder}/kernel.asmgen -o:{outFolder}/kernel.o'
FIRST_PHASE_LINK_COMMAND_TEMPLATE = '-T {firstPhaseScript} -Ur -O9 -EL {outFolder}/kernel.o  \
     --start-group {mvToolsDir}/common/moviCompile/lib/ma2x8x/mlibc_lite.a {mvToolsDir}/common/moviCompile/lib/ma2x8x/mlibc_lite_lgpl.a \
         {mvToolsDir}/common/moviCompile/lib/ma2x8x/mlibm.a {mvToolsDir}/common/moviCompile/lib/ma2x8x/mlibcrt.a --end-group \
             -o {outFolder}/kernel.mvlib -Map {outFolder}/kernel.mvlib.map'
SECOND_PHASE_LINK_COMMAND_TEMPLATE = '-EL -s -T {secondPhaseScript} {outFolder}/kernel.mvlib -o {outFile} -Map {outFolder}/kernel.mvlib.map'

TEMP_FOLDER_NAME        = 'build'
DEFAULT_LDSCRIPTS_PATH  = './ldscripts'
FIRST_PHASE_SCRIPT_NAME = 'shave_first_phase.ld'
LDSCRIPT_NAME           = 'myriad2_dynamic_shave_slice_computeaorta.ldscript'

# commands

MOVI_COMPILE_COMMAND      = ''
MOVI_ASM_COMMAND          = ''
FIRST_PHASE_LINK_COMMAND  = ''
SECOND_PHASE_LINK_COMMAND = ''

def setEnvironment(args):
    global MOVI_COMPILE_COMMAND
    global MOVI_ASM_COMMAND
    global FIRST_PHASE_LINK_COMMAND
    global SECOND_PHASE_LINK_COMMAND

    sourceDir = os.path.dirname(os.path.abspath(__file__)) + '/'

    pathToLdscripts = args.scripts  if args.scripts != '' else DEFAULT_LDSCRIPTS_PATH
    pathToOutputFile = args.output if args.output != '' else os.path.splitext(args.input)[0]+'.elf'

    pathToOutTempFolder = os.path.join(sourceDir, TEMP_FOLDER_NAME)
    pathFirstPhaseScript = os.path.join(pathToLdscripts, FIRST_PHASE_SCRIPT_NAME)
    pathSecondPhaseScript = os.path.join(pathToLdscripts, LDSCRIPT_NAME)

    if not os.path.isabs(pathToOutputFile) :
        pathToOutputFile = sourceDir + pathToOutputFile

    os.makedirs(os.path.dirname(pathToOutputFile), exist_ok=True)
    os.makedirs(pathToOutTempFolder, exist_ok=True)

    MOVI_COMPILE_COMMAND      = MOVI_COMPILE_COMMAND_TEMPLATE.format(pathToKernel=args.input, outFolder=pathToOutTempFolder)
    MOVI_ASM_COMMAND          = MOVI_ASM_COMMAND_TEMPLATE.format(outFolder=pathToOutTempFolder)
    FIRST_PHASE_LINK_COMMAND  = FIRST_PHASE_LINK_COMMAND_TEMPLATE.format(firstPhaseScript=pathFirstPhaseScript, outFolder=pathToOutTempFolder, mvToolsDir=args.tools)
    SECOND_PHASE_LINK_COMMAND = SECOND_PHASE_LINK_COMMAND_TEMPLATE.format(secondPhaseScript=pathSecondPhaseScript, outFolder=pathToOutTempFolder, outFile=pathToOutputFile)

def run_tool(tool, args):
    isCompleted = False
    command = f'{tool} {args}'
    absDir = os.path.join(os.getcwd(), 
                            os.path.dirname(os.path.abspath(__file__)))

    try:
        process = subprocess.Popen([command], stderr=subprocess.PIPE, shell=True, cwd=absDir)
        isCompleted = process.wait() == 0
    except subprocess.CalledProcessError as ex:
        _, err = process.communicate()
        raise Exception(command, err)
    except:
        raise Exception(command, f'Unexpected error: {sys.exc_info()[0]}\n')

    if isCompleted is False:
        _, err = process.communicate()
        raise Exception(command, err)

parser = argparse.ArgumentParser(
    description='Build .cpp kernel source file to execute it on MyriadX device')
parser.add_argument('--i', dest='input',required=True,
                        help='path to source file')
parser.add_argument('--t', dest='tools',required=True,
                        help='path to tools')
parser.add_argument('--ld', dest='scripts', default='',
                        help=f'path to ldscripts folder (default: {DEFAULT_LDSCRIPTS_PATH})')
parser.add_argument('--o', dest='output', default='',
                        help=f'output file name')

args = parser.parse_args()
setEnvironment(args)

try:
    run_tool(f'{args.tools}/linux64/bin/moviCompile', MOVI_COMPILE_COMMAND)
    run_tool(f'{args.tools}/linux64/bin/moviAsm', MOVI_ASM_COMMAND)
    run_tool(f'{args.tools}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld', FIRST_PHASE_LINK_COMMAND)
    run_tool(f'{args.tools}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld', SECOND_PHASE_LINK_COMMAND)
except Exception as ex:
    command, stderr = ex.args
    print(f'Build failed.\nCommand line:\n{command}\nError:\n{stderr}\n')
    exit(1)    
