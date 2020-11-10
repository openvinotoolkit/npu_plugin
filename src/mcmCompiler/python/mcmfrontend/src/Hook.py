#!/usr/bin/env python

"""
Forked From https://gist.github.com/spulec/1364640#file-pre-commit
"""

import os
import re
import subprocess
import sys

# modified = re.compile('^[MA]\s+(?P<name>.*)$')
modified = re.compile(r"[MA]+\s+(?P<name>.*)$")  # nopep8

CHECKS = [
    {
        'output': 'Checking for print statements...',
        'command': 'grep -n "^\s*print" %s',
        'match_files': ['.*\.py$'],
        'ignore_files': [],
        'allow_error':True,
        'print_filename': True,
        'exists': False,
        'package': ''
    },
    {
        'output': 'Checking for TODO statements...',
        'command': 'grep -n "^\s*TODO" %s',
        'match_files': ['.*\.py$'],
        'ignore_files': [],
        'allow_error':True,
        'print_filename': True,
        'exists': False,
        'package': ''
    },
    {
        'output': 'Running Pyflakes...',
        'command': 'pyflakes %s',
        'match_files': ['.*\.py$'],
        'ignore_files': [],
        'allow_error':False,
        'print_filename': True,
        'exists': True,
        'package': 'pyflakes'
    },
    {
        'output': 'Running pep8...',
        'command': 'pycodestyle -r --ignore=E501,W293,W605,W503,W504 %s',
        'match_files': ['.*\.py$'],
        'ignore_files': [],
        'allow_error':False,
        'print_filename': False,
        'exists': True,
        'package': 'pycodestyle'
    }
]


def highlight(text, status):
    attrs = []
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
    }
    if not sys.stdout.isatty():
        return text
    attrs.append(colors.get(status, 'red'))
    attrs.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attrs), text)


def exists(cmd, error=True):
    devnull = open(os.devnull, 'w')
    params = {'stdout': devnull, 'stderr': devnull, }
    query = 'which %s' % cmd
    code = subprocess.call(query.split(), **params)
    if code != 0 and error:
        print(highlight('not installed %(command)s' % {'command': cmd}, 'yellow'))
        sys.exit(1)


def matches_file(file_name, match_files):
    return any(
        re.compile(match_file).match(file_name) for match_file in match_files)


def system(*args, **kwargs):
    kwargs.setdefault('stdout', subprocess.PIPE)
    proc = subprocess.Popen(args, **kwargs)
    out, err = proc.communicate()
    return out, err


def check_files(files, check):
    result = 0
    print(highlight(check['output'], 'yellow'))

    if check['exists'] and check['package']:
        exists(check['package'])

    for file_name in files:
        if 'match_files' not in check or matches_file(file_name,
                                                      check['match_files']):
            if 'ignore_files' not in check or not matches_file(
                    file_name, check['ignore_files']):
                out, err = system(check['command'] % file_name,
                                  stderr=subprocess.PIPE, shell=True)
                if out or err:
                    if check['print_filename']:
                        prefix = '\t%s:' % file_name
                    else:
                        prefix = '\t'
                    output_lines = ['%s%s' % (prefix, line) for line in
                                    out.splitlines()]

                    if not check['allow_error']:
                        print(highlight('\n'.join(output_lines), 'red'))
                    else:
                        print(highlight('\n'.join(output_lines), 'cyan'))

                    if err:
                        print(highlight(err, 'red'))

                    if not check['allow_error']:
                        result = 1
    return result


def main():
    # Stash any changes to the working tree that are not going to be committed
    # subprocess.call(['git', 'stash', '-u', '--keep-index'],
    #                stdout=subprocess.PIPE)

    files = []
    out, err = system('git', 'status', '--porcelain')
    for line in out.splitlines():
        line = line.decode('ascii')
        # print("Line:", line, end="")
        match = modified.match(str(line))
        # print(match)
        if match:
            files.append(match.group('name'))

    result = 0

    for check in CHECKS:
        result = check_files(files, check) or result

    # # Unstash changes to the working tree that we had stashed
    # subprocess.call(['git', 'reset', '--hard'], stdout=subprocess.PIPE,
    #                 stderr=subprocess.PIPE)
    # subprocess.call(['git', 'stash', 'pop', '--quiet', '--index'],
    #                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sys.exit(result)


if __name__ == '__main__':
    main()
