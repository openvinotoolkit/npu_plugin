import os
import subprocess
import stat


def get_git_folder():
    cmd = "git rev-parse --git-path hooks"
    cmd = cmd.split(" ")
    return subprocess.check_output(cmd).decode("ascii").strip()


def exists_already(f1, f2):

    if not os.path.exists(f1):
        return False

    f = open(f1, "r")
    a = f.readlines()
    f.close()

    f = open(f2, "r")
    b = f.readlines()
    f.close()

    a_cc = 0
    for l in a:
        for c in l:
            a_cc += 1

    b_cc = 0
    for l in b:
        for c in l:
            b_cc += 1

    if a_cc != b_cc:
        return False

    if(stat.S_IMODE(os.lstat(f1).st_mode) & 511 == 511):
        return True
    else:
        return False


def setup_hooks():
    path = get_git_folder()
    hookfile = os.path.join(path, "pre-commit")
    pyfile = "Hook.py"
    pyfile = os.path.join(os.path.dirname(__file__), pyfile)

    if exists_already(hookfile, pyfile):
        print("Githooks detected.", hookfile)
        return 0

    try:
        f = open(pyfile)
        hook = f.read()
        f.close()
    except FileNotFoundError:
        print("Githook install error: ", pyfile, "not found")
        return -1

    try:
        f = open(hookfile, "w")
        f.write(hook)
        f.close()
    except FileNotFoundError:
        print("Githook install error: ", hookfile, "not found")
        return -1

    try:
        os.chmod(hookfile, 0o777)
    except PermissionError:
        return -1


if __name__ == "__main__":
    exitcode = setup_hooks()
    if exitcode == -1:
        print("Warning: Cannot install git hooks.")
        print("Please run Environment.py with sudo")
