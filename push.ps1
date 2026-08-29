$comment = Read-Host "Enter commit comment"

git add .
git commit -m "$comment"
git push