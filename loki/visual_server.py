# loki/visual_server.py

import uvicorn
import logging
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Настройка базового логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Определяем допустимые состояния через Enum для строгой валидации
class StatusEnum(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"


# Определяем модель данных для входящего запроса
class State(BaseModel):
    status: StatusEnum


# Создаем экземпляр FastAPI
app = FastAPI(title="LOKI Visual Server")


@app.post("/set_state")
async def set_state(state: State):
    """
    Принимает POST-запрос для обновления состояния.
    Валидация выполняется автоматически с помощью Pydantic и StatusEnum.
    """
    logging.info(f"Received new state: {state.status.value}")
    return {"message": f"State successfully set to {state.status.value}"}


@app.get("/")
async def root():
    return {"message": "LOKI Visual Server is running."}


if __name__ == "__main__":
    logging.info("Starting LOKI Visual Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
