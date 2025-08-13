# Redact-LLM
Real-time LLM Red Teaming Lab. Stress test your prompts, uncover hallucinations, and expose vulnerabilities with AI-powered attacks. Built on FastAPI + Redis Streams.


Running locally, firstly fill both .env files in /frontend and /backend as per the corresponding .env.example files

Secondly navigate to the correct directories:

cd frontend

npm install

npm run dev

then to set up backend, go back to the root directory, then:

cd backend

pip install requirements.txt

uvicorn main:app --reload --port 8000

head over to: http://localhost:8080 and your local development server is successfully running!
