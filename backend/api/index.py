
from main import app

# This is the handler function that Vercel will call
def handler(request):
    return app(request.scope, request.receive, request.send)

# Export for Vercel
app = app
