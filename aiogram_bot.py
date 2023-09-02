from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.dispatcher.filters.state import State, StatesGroup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import logging

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)
storage = MemoryStorage()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=storage)

class Form(StatesGroup):
    name = State()

@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await Form.name.set()
    await message.reply("Hi! Send me a PDF file.")

@dp.message_handler(content_types=['document'], state=Form.name)
async def process_pdf(message: types.Message, state: FSMContext):
    pdf = await bot.download_file_by_id(message.document.file_id, destination='yourfile.pdf')
    
    pdf_reader = PdfReader('yourfile.pdf')
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    await bot.send_message(
        message.chat.id,
        "Your processed information here",
        parse_mode=ParseMode.HTML,
    )
    await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
