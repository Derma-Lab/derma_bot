# Dermalab AI Chatbot

## Prerequisites
- Ensure you have Docker and Docker Compose installed.
- For local development, ensure Node.js and Python 3.12 are installed.


### Running with Docker
1. Build and start the Docker container:
   ```bash
   docker-compose up
   ```
2. Access the frontend at `http://localhost:3000`.
3. Access the backend at `http://localhost:4000`.


### Running Frontend Locally
1. Navigate to the `ai-chatbot` directory.
2. Install dependencies:
   ```bash
   pnpm install
   ```
3. Start the development server:
   ```bash
   pnpm dev
   ```
4. Access the frontend at `http://localhost:3000`.

### Running Backend Locally
1. Navigate to the `backend` directory.
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 4000 --reload
   ```
5. The backend will be available at `http://localhost:4000`.

## Environment Variables
- Ensure to configure the `.env` files for both frontend and backend with necessary environment variables.

## Database Migrations
- To run database migrations, use:
  ```bash
  pnpm run db:migrate
  ```

## Notes
- Ensure all services are up and running before accessing the application.
- Adjust any configurations as necessary in the `.env` files.
