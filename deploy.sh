#!/bin/bash
# Railway Deployment Helper Script

echo "🚀 Meeting Captioning Studio - Railway Deployment"
echo "=================================================="
echo

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
    echo "✅ Railway CLI installed"
else
    echo "✅ Railway CLI found"
fi

echo

# Login to Railway
echo "🔑 Logging in to Railway..."
railway login

echo

# Deploy the application
echo "🚀 Deploying to Railway..."
echo "   This may take a few minutes..."
echo

railway up

echo
echo "🎉 Deployment complete!"
echo
echo "📋 Next steps:"
echo "1. Go to your Railway dashboard"
echo "2. Set up environment variables in the Variables tab"
echo "3. Copy the deployment URL"
echo "4. Test your application!"
echo
echo "🌐 Your app should be available at:"
echo "   https://your-app-name.railway.app"
echo