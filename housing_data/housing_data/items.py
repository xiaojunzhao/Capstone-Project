# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class HousingDataItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    
    # Address, Postal Code, Housing area, Housing Price, Baths, Beds, Built year, Housing type
    #link = scrapy.Field()
    Address = scrapy.Field()
    Postal_code = scrapy.Field()
    Housing_area = scrapy.Field()
    Housing_price = scrapy.Field()
    #main_feature = scrapy.Field()
    #room = scrapy.Field()
    Baths = scrapy.Field()
    Beds = scrapy.Field()
    Built_year = scrapy.Field()
    Housing_type = scrapy.Field()
    HOA = scrapy.Field()
    Crime_level = scrapy.Field()
    
