@echo off
REM Railway Deployment Helper Script for Windows

echo 🚀 Meeting Captioning Studio - Railway Deployment
echo ==================================================
echo.

REM Check if Railway CLI is installed
railway --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Railway CLI not found. Installing...
    npm install -g @railway/cli
    echo ✅ Railway CLI installed
) else (
    echo ✅ Railway CLI found
)

echo.

REM Login to Railway
echo 🔑 Logging in to Railway...
railway login

echo.

REM Deploy the application
echo 🚀 Deploying to Railway...
echo    This may take a few minutes...
echo.

railway up

echo.
echo 🎉 Deployment complete!
echo.
echo 📋 Next steps:
echo 1. Go to your Railway dashboard
echo 2. Set up environment variables in the Variables tab
echo 3. Copy the deployment URL
echo 4. Test your application!
echo.
echo 🌐 Your app should be available at:
echo    https://your-app-name.railway.app
echo.