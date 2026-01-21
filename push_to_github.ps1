Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Pushing Churn Prediction System to GitHub" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Write-Host "`n1. Initializing Git repository..." -ForegroundColor Yellow
git init

Write-Host "`n2. Adding files..." -ForegroundColor Yellow
git add .

Write-Host "`n3. Committing files..." -ForegroundColor Yellow
git commit -m "Initial commit of Membership Churn Prediction System"

Write-Host "`n4. Adding remote origin..." -ForegroundColor Yellow
# Remove existing origin if it exists to avoid errors
git remote remove origin 2>$null
git remote add origin https://github.com/dataverseanalytics/Churn_Prediction.git

Write-Host "`n5. Renaming branch to main..." -ForegroundColor Yellow
git branch -M main

Write-Host "`n6. Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " Done! Check the output above for any errors." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
