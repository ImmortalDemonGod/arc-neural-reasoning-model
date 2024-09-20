#!/usr/bin/env python3
import subprocess
import re

def get_branches():
    result = subprocess.run(['git', 'branch'], capture_output=True, text=True)
    return result.stdout.splitlines()

def delete_branch(branch):
    subprocess.run(['git', 'branch', '-D', branch])

def main():
    branches = get_branches()
    pattern = re.compile(r'^\s*fix-.*-[0-9a-f]{8}$')
    
    for branch in branches:
        if pattern.match(branch.strip()):
            print(f"Deleting branch: {branch.strip()}")
            delete_branch(branch.strip())

if __name__ == "__main__":
    main()
