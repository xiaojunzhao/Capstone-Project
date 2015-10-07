#import sys

#sys.path.append('C:\\Users\\Xiaojun\\Desktop\\hupulinks')

import scrapy
from housing_data.items import HousingDataItem
from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
import re


class housingdataSpider(CrawlSpider):
    name = "housingdata"
    allowed_domains=["trulia.com"]
    base_url = "http://www.trulia.com/for_sale/New_York,NY"
    start_urls = []
    for i in range(1,401):   
        start_url = base_url+"/"+str(i)+"_p"
        start_urls.append(start_url)

    rules = ( Rule(LinkExtractor(allow=('\/property\/.+',),
              restrict_xpaths=('//a[@class = "primaryLink pdpLink activeLink"]')),
             callback="parse_items",follow=True),)
    
    def parse_items(self, response):
        item=HousingDataItem()
        
        #item['main_feature'] = response.xpath('//ul[@class = "listBulleted listingDetails mrn mtm list3cols"]\/li/text()').extract()[:-1]
        main_feature = response.xpath('//ul[@class = "listBulleted listingDetails mrn mtm list3cols"]    \
                                     /li/text()').extract()[:-1]
        
        item["Housing_type"] = main_feature[0]
        feature_str = ','.join(main_feature)
        if "Bedroom" in feature_str:
            item['Beds'] = ''.join(re.findall('\d+ Bedroom',feature_str))
        else: item['Beds'] = "NA"

        if "Bathroom" in feature_str:
            item['Baths'] = ''.join(re.findall('\d.+full.+Bathroom',feature_str))
        else: item['Baths'] = "NA"
        
        if "sqft" in feature_str:
            item['Housing_area'] = ''.join(re.findall('\d+ sqft|\d+\,\d+ sqft',feature_str))
        else: item['Housing_area'] = "NA"

        if "/sqft" in feature_str:
            item['Housing_price'] = ''.join(re.findall('\$\d+/\sqft|\$\d+\,\d+\/sqft',feature_str))
        else: item["Housing_price"] = "NA"

        if "Built" in feature_str:
            item['Built_year'] = ''.join(re.findall('Built in \d+',feature_str))
        else: item['Built_year'] = "NA"
        item['Address'] = response.xpath('//span[@class = "headingDoubleSuper h2 typeWeightNormal mvn ptn"]/text()').extract()
        item['Postal_code'] = response.xpath('//span[@itemprop = "postalCode"]/text()').extract()
        if response.xpath('//span[@class = "plm"]/text()'):
            item['HOA'] = response.xpath('//span[@class = "plm"]/text()').extract()[0].strip().split('\n')[1].strip()
        else: item['HOA'] = "NA"
        if response.xpath('//div[@class = "h5 typeWeightNormal mediaImg mediaImgExt typeHighlight"]/text()'):
            item['Crime_level'] = response.xpath('//div[@class = "h5 typeWeightNormal mediaImg mediaImgExt typeHighlight"]/text()').extract()[0].strip()
        else: item['Crime_level'] = "NA"           
        yield item
        


    

    

    
              
