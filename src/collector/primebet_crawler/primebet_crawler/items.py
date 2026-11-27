# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class PromotionItem(scrapy.Item):
    title = scrapy.Field()
    category = scrapy.Field()      # ex: Free Bet, Cashback, Acca Boost, Other
    summary = scrapy.Field()
    url = scrapy.Field()
    is_free_bet = scrapy.Field()   # True si le titre ou résumé mentionne "free bet"


class FootballMatchItem(scrapy.Item):
    competition = scrapy.Field()
    home_team = scrapy.Field()
    away_team = scrapy.Field()
    kickoff_time = scrapy.Field()
    odds = scrapy.Field()
    

class CasinoGameItem(scrapy.Item):
    name = scrapy.Field()
    category = scrapy.Field()
    min_stake_usd = scrapy.Field()
    max_win_usd = scrapy.Field()
    priority = scrapy.Field()
    