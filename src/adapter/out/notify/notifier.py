import telepot
import configuration

def notify(msg):
    bot = telepot.Bot(configuration.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendMessage(configuration.TELEGRAM_TO, msg, parse_mode='Markdown')
