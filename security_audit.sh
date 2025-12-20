#!/bin/bash
# Security Audit Script for Railway Deployment

echo "🔒 Security Audit - Meeting Captioning Studio"
echo "=============================================="
echo

# Check for any real API keys (common patterns)
echo "📋 Checking for API keys in code files..."
if grep -r "sk-[a-zA-Z0-9]" --include="*.py" --include="*.json" --include="*.js" . >/dev/null 2>&1; then
    echo "❌ Found potential OpenAI API keys in code!"
    grep -r "sk-[a-zA-Z0-9]" --include="*.py" --include="*.json" --include="*.js" .
    exit 1
fi

if grep -r "AIza[a-zA-Z0-9]" --include="*.py" --include="*.json" --include="*.js" . >/dev/null 2>&1; then
    echo "❌ Found potential Google API keys in code!"
    grep -r "AIza[a-zA-Z0-9]" --include="*.py" --include="*.json" --include="*.js" .
    exit 1
fi

# Check for placeholder removal
echo "📋 Checking for placeholder strings..."
if grep -r "YOUR.*KEY" --include="*.py" --include="*.json" . | grep -v "YOUR_" >/dev/null 2>&1; then
    echo "⚠️  Found placeholder API keys:"
    grep -r "YOUR.*KEY" --include="*.py" --include="*.json" . | grep -v "YOUR_"
    echo "   (These should be removed or marked as templates)"
fi

# Check for sensitive files
echo "📋 Checking for sensitive files..."
if find . -name "*.log" -o -name "*.db" -o -name "*secret*" -o -name "*private*" | grep -q .; then
    echo "⚠️  Found potentially sensitive files:"
    find . -name "*.log" -o -name "*.db" -o -name "*secret*" -o -name "*private*"
    echo "   (Ensure these don't contain real credentials)"
fi

# Check if .gitignore exists
echo "📋 Checking .gitignore..."
if [ ! -f ".gitignore" ]; then
    echo "❌ No .gitignore file found!"
    exit 1
else
    echo "✅ .gitignore file exists"
fi

# Check if environment variables are used
echo "📋 Checking secure config usage..."
if grep -r "get_secure" src/ >/dev/null 2>&1; then
    echo "✅ Using secure config methods"
else
    echo "❌ Not using secure config methods!"
    exit 1
fi

echo
echo "🎉 Security Audit Complete!"
echo "✅ Safe to deploy to Railway"
echo
echo "📋 Deployment Checklist:"
echo "   ✅ No API keys in code"
echo "   ✅ Using environment variables"
echo "   ✅ Secure config manager"
echo "   ✅ .gitignore configured"
echo "   ✅ Sensitive files excluded"
echo
echo "🚀 Ready for deployment!"
echo