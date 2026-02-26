#!/bin/bash
echo "🔍 Checking deployment setup..."

# Check static folder
if [ -d "static" ]; then
    echo "✅ Static folder exists"
    if [ -f "static/style.css" ]; then
        echo "✅ style.css exists"
    else
        echo "❌ style.css missing!"
    fi
else
    echo "❌ Static folder missing!"
fi

# Check templates
if [ -d "templates" ]; then
    echo "✅ Templates folder exists"
    if [ -f "templates/chat.html" ]; then
        echo "✅ chat.html exists"
    else
        echo "❌ chat.html missing!"
    fi
else
    echo "❌ Templates folder missing!"
fi

# Check src
if [ -d "src" ]; then
    echo "✅ Src folder exists"
    for file in helper.py prompt.py; do
        if [ -f "src/$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing!"
        fi
    done
else
    echo "❌ Src folder missing!"
fi

echo ""
echo "📝 To fix static files:"
echo "1. Make sure static/style.css exists"
echo "2. Use {{ url_for('static', filename='style.css') }} in HTML"
echo "3. Check Flask app has static_folder='static'"