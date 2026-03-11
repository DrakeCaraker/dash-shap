#!/bin/bash

echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

git rm -r --cached **/__pycache__/ 2>/dev/null || true

git add .gitignore
git commit -m "Remove __pycache__ from tracking and add to .gitignore"
git push origin main

echo "Done."
