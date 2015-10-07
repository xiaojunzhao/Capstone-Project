# -*- coding: cp936 -*-
import os
import regression_import as ri
import json
import urllib2
import xmltodict
import numpy as np
import csv
import re
import pandas as pd

from bokeh.browserlib import view
from bokeh.document import Document
from bokeh.embed import file_html     # BOKEH can generate standalone HTML documents
from bokeh.models.glyphs import Circle
from bokeh.models import (
    GMapPlot, Range1d, ColumnDataSource, LinearAxis,
    PanTool, WheelZoomTool, BoxSelectTool, HoverTool,
    BoxSelectionOverlay, GMapOptions,
    NumeralTickFormatter, PrintfTickFormatter
)
from bokeh.resources import INLINE    #provide minified bokehjs from the static directory
from bokeh.plotting import show
from bokeh.plotting import figure
from flask import Flask, render_template,request
from bokeh.embed import components
from bokeh.charts import Bar, output_file, show
from geopy.geocoders import Nominatim
from bokeh.charts import defaults


from sklearn import preprocessing
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import grid_search
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor



input_file = csv.DictReader(open('out.csv'))
input_file_1 = csv.DictReader(open('page.csv'))


input_file_list = []
for row0 in input_file:
    input_file_list.append(row0) 

input_file1_list=[]
for row1 in input_file_1:
    input_file1_list.append(row1)


new_list = []
s =0
for item1 in input_file1_list:
    for item0 in input_file_list:
        if item1['Address'] == item0['Address']:
            crime_level = item1['Crime_level']
            #print crime_level
            item0['Crime_level'] = crime_level
            new_list.append(item0)
housing_data_list = []          
for row in new_list:    
    if row['Housing_area'] == "NA" or row['Housing_price'] == "NA":
        continue
    else:
        if row['Crime_level'] =="NA":
            row['Crime_level'] = "High"
        crime = row['Crime_level']
        location = row['Location'].split(',')
        latitude = location[0]
        longitude = location[1]
        if row['Beds'] !="NA":
            beds = float(row['Beds'].replace('bd',''))
        else: beds = ''
        if row['Baths'] !="NA":
            baths = float(row['Baths'].replace('ba',''))
        else: baths = ''
        if row['Housing_price'] != "NA" and row['Housing_price'] != "":
            housing_price =float(re.sub(r'[/$/,/+]','',row['Housing_price']))
        else: housing_price = ''
        if row['Housing_area'] != "NA":
            housing_area =float(re.sub(r'[sqft/,/+]','',row['Housing_area']))
        else: housing_area = ''
        if housing_area and housing_price:
            housing_price_area = "$"+"{0:.2f}".format(housing_price/housing_area)+"/sqft"
            item = (row['Address'],row['ZipCode'],latitude,longitude,\
                row['Housing_type'],beds,baths,\
                housing_price,housing_area, housing_price_area,crime)
                #print item[5]
            housing_data_list.append(item)
housing_data_list = list(set(housing_data_list))
#print housing_data_list

app = Flask(__name__)

@app.route("/",methods = ['POST','GET'])
def home():  
    housing_type_dict = {'condo':1,'coop':2,'Single-Family Home':3,'loft':4,'townhome':5,'Multi-Family Home':6,'Apt/Condo/Twnhm':7,
                     'apartment':8,'Unspecified property type':9}
    if request.method == 'POST':    
        address = request.form['address_value']
        
        geolocator = Nominatim()
        location = geolocator.geocode(address)
        longitude = location.longitude
        latitude = location.latitude

        # Find the closest 25 housing items in the database
        ri.df_final['distance'] = ri.df_final['Latitude'].apply(np.square)+ri.df_final['Longitude'].apply(np.square)
        x = longitude*longitude+latitude*latitude
        selected_data = ri.df_final.iloc[np.argsort((ri.df_final.distance-x).abs())[0:50]]
        final_selected_data = selected_data.loc[selected_data['Housing_price']<8000]

        # feature concatenation
        cat_features = final_selected_data[['Housing_type']].values.tolist()
        enc = preprocessing.OneHotEncoder()
        enc.fit(cat_features)  
        cat_feature_transform = enc.transform(cat_features).toarray()
        num_feature = final_selected_data[['Latitude','Longitude','Beds','Baths']].values
        total_feature = np.hstack((num_feature,cat_feature_transform))

        #total_feature = num_feature
        X_train = total_feature
        Y_train = final_selected_data['Housing_price'].values[0:]
        #neigh = KNeighborsRegressor(weights ='distance',n_neighbors=3,algorithm = 'kd_tree')
        #parameters = {'leaf_size':range(1,41),'metric':['minkowski'],'n_neighbors':range(1,3)}
        #clf = grid_search.GridSearchCV(neigh, parameters,cv=2)
        #clf.fit(X_train, Y_train)
        #best_estimator = clf.best_estimator_ 

        rf = RandomForestRegressor(n_estimators=20,criterion='mse',max_features ="sqrt")
        parameters = {'n_estimators':range(1,25),'max_features':['auto','sqrt','log2']}
        clf_rf = grid_search.GridSearchCV(rf, parameters,cv=3)
        clf_rf.fit(X_train, Y_train)
        best_estimator_rf = clf_rf.best_estimator_ 


        # transform the input data 
        housing_type = request.form["Housing_type"]
        bed = float(request.form['#Bedroom'])
        bath = float(request.form['#Bathroom'])
        type_cat = housing_type_dict[housing_type]
        try:
            cat_test = enc.transform([type_cat]).toarray()
            sub_df = pd.DataFrame([[latitude,longitude,bed,bath]],columns=["latitude",'longitude','bed','bath'])
            num_test = sub_df[['latitude','longitude','bed','bath']].values
            X_test = np.hstack((num_test,cat_test))
            prediction_price = best_estimator_rf.predict(X_test).tolist()[0]
            prediction = "$"+"{0:.2f}".format(prediction_price)+"/sqft"     
        except:
            prediction = "unable to find the similar housing type nearby" 
    else:
        prediction = "None"
    #prediction = "The address you entered cannot be encoded"
    return render_template('home.html',prediction = prediction)



@app.route("/index",methods = ["POST","GET"])
def index():
    # Choose from zipcode
    zipcode = request.form['zip code']
    region_housing_data_list = []
    bed_filter = []
    region_housing_lo = []
    region_housing_la = []
    region_housing_price = []
    region_house_address = []
    count = 0
    sum_lo = 0.0
    sum_lat = 0.0
    for item in housing_data_list:     
        if item[1] == zipcode:
            count+=1
            bed_filter.append(item)
            region_housing_data_list.append(item)
            region_housing_lo.append(item[3]) 
            region_housing_la.append(item[2])
            region_housing_price.append(item[9])
            region_house_address.append(item[0])
            sum_lo = sum_lo+float(item[3])
            sum_lat = sum_lat+float(item[2])
        elif zipcode=="":
            count+=1
            bed_filter.append(item)
            region_housing_data_list.append(item)
            region_housing_lo.append(item[3]) 
            region_housing_la.append(item[2])
            region_housing_price.append(item[9])
            region_house_address.append(item[0])
            sum_lo = sum_lo+float(item[3])
            sum_lat = sum_lat+float(item[2])

    # Choose from number of bedroom
    num_beds = request.form['bedroom']
    if num_beds =='0':
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in region_housing_data_list:
            count+=1
            bed_filter.append(item)
            region_housing_lo.append(item[3]) 
            region_housing_la.append(item[2])
            region_housing_price.append(item[9])
            region_house_address.append(item[0])
            sum_lo = sum_lo+float(item[3])
            sum_lat = sum_lat+float(item[2])        
    if num_beds == '1':
        bed_filter = []
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in region_housing_data_list:
            if item[5] == 1:
                count+=1
                bed_filter.append(item)
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])
    elif num_beds == '2':
        bed_filter = []
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in region_housing_data_list:
            if item[5] == 2:
                count+=1
                bed_filter.append(item)
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])
    elif num_beds == '3+':
        bed_filter = []
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in region_housing_data_list:
            if item[5] >=3:
                count+=1
                bed_filter.append(item)
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])


    ## Choose from price range
    price_range = request.form['price_range']
    if price_range =="none":
        pass
    if price_range == "0-1000":
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in bed_filter:
            price = re.sub(r'[^0-9\.]','',item[9])
            if float(price) <=1000:
                count+=1
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])
    if price_range == "1000-2000":
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in bed_filter:
            price = re.sub(r'[^0-9\.]','',item[9])
            if float(price)>1000 and float(price) <=2000:
                count+=1
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])
    if price_range == "2000-3000":
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in bed_filter:
            price = re.sub(r'[^0-9\.]','',item[9])
            if float(price)>2000 and float(price) <=3000:
                count+=1
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])
    if price_range == "3000+":
        region_housing_lo = []
        region_housing_la = []
        region_housing_price = []
        region_house_address = []
        count = 0
        sum_lo = 0.0
        sum_lat = 0.0
        for item in bed_filter:
            price = re.sub(r'[^0-9\.]','',item[9])
            if float(price) > 3000:
                count+=1
                region_housing_lo.append(item[3]) 
                region_housing_la.append(item[2])
                region_housing_price.append(item[9])
                region_house_address.append(item[0])
                sum_lo = sum_lo+float(item[3])
                sum_lat = sum_lat+float(item[2])



    
    ##Drawing a google map
    if count == 0:
        region_lat = 40.7127
        region_lon = -74.0059
    else:
        region_lat = sum_lat/count
        region_lon = sum_lo/count   
    y_range = Range1d()  #setting ranges of the graph
    x_range = Range1d()

    map_options = GMapOptions(lat=region_lat, lng=region_lon, map_type="roadmap",zoom=13,styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""" )

    plot = GMapPlot(
        x_range=x_range, y_range=y_range,
        map_options=map_options,
        title = "NYC Housing Search"
    )

    source = ColumnDataSource(
        data=dict(
            lon=region_housing_lo,
            lat=region_housing_la,
            Housing_Price=region_housing_price,
            Address = region_house_address,
            fill = ['blue']*len(region_housing_la)
        )
    )

    # Mouse over the markers
   
    hover = HoverTool(
        tooltips=[
            ("Address","@Address"),
            ("Housing Price", "@Housing_Price"),
            ("location(lon,lat)", "($x, $y)"),
        ]
    )
   

    circle = Circle(x="lon", y="lat", size=7, fill_color="fill", line_color="blue",fill_alpha=0.6)
    plot.add_glyph(source, circle)

    pan = PanTool()
    wheel_zoom = WheelZoomTool()
    box_select = BoxSelectTool()

    plot.add_tools(pan, wheel_zoom, box_select,hover)

    xaxis = LinearAxis(axis_label = "lon",major_tick_in = 0, formatter=NumeralTickFormatter(format='0.0000'))
    plot.add_layout(xaxis,'below')

    yaxis = LinearAxis(axis_label = 'lat',major_tick_in = 0, formatter = PrintfTickFormatter(format="%.4f"))
    plot.add_layout(yaxis, 'left')


    overlay = BoxSelectionOverlay(tool=box_select)  # An overlay renderer that Tool objects can use to render a ¡®rubber band¡¯ selection box on a Plot.
    plot.add_layout(overlay)

    script, div = components(plot)

    return render_template('google_map.html',script = script, div = div)



crime_dict = {"High":4,"Moderate":3,"Low":2,"Lowest":1}
new_housing_data_list = [list(l) for l in housing_data_list]
crime_level_list = []
housing_price_list = []
for item in new_housing_data_list:
    price = price = re.sub(r'[^0-9\.]','',item[9])
    item[10] = crime_dict[item[10]]
    crime_level_list.append(item[10])
    housing_price_list.append(float(price))
@app.route("/color_depth_map",methods = ["POST","GET"])
def depth():
    p = figure(plot_width=520, plot_height=520)
    p.circle(crime_level_list, housing_price_list, size=15, color="navy", alpha=0.5)
    script, div = components(p)
    return render_template('color_depth.html',script = script, div = div)




@app.route('/trend',methods = ["POST"])
def trend():
    zipcode = request.form['zip code']
    url = "http://api.trulia.com/webservices.php?library=TruliaStats&function=getZipCodeStats&zipCode="+str(zipcode)+"&startDate=2010-01-01&endDate=2015-02-01&apikey=69u53ycaa4he2guamqv8n4ny"
    xml_file = urllib2.urlopen(url)
    data=xml_file.read()
    data = xmltodict.parse(data)
    count=0
    weekending_list=[]
    date_list=[]    
    bed2_med_price_list=[]
    bed3_med_price_list=[]
    bed1_med_price_list=[]
    df = pd.DataFrame()
    for item in data['TruliaWebServices']['response']['TruliaStats']['listingStats']['listingStat']:
    
        weekending_list.append(str(item['weekEndingDate']))
        bed1_med_price=item['listingPrice']['subcategory'][1]['medianListingPrice']
        bed2_med_price=item['listingPrice']['subcategory'][2]['medianListingPrice']
        bed3_med_price=item['listingPrice']['subcategory'][3]['medianListingPrice']

   
        bed1_med_price_list.append(float(bed1_med_price))
        bed2_med_price_list.append(float(bed2_med_price))
        bed3_med_price_list.append(float(bed3_med_price))
        
   
    df['Date'] = weekending_list
    df['1_bed'] = bed1_med_price_list
    df['2_bed'] = bed2_med_price_list
    df['3_bed'] = bed3_med_price_list    
    df['Date'] = pd.to_datetime(df.Date,format = "%Y-%m-%d")

    p1 = figure(plot_width=520, plot_height=520,x_axis_type="datetime")
    p1.line(df['Date'], df['1_bed'],color="orange", alpha=0.8)
    p1.title = "1 Bed Listing Price Trend in "+zipcode
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Median Price'
    p1.ygrid.band_fill_color="olive"
    p1.ygrid.band_fill_alpha = 0.1
    p1.title_text_font_style = "italic"
    p1.yaxis[0].formatter = NumeralTickFormatter(format="$0,0.00")
    script1, div1 = components(p1)

    p2 = figure(plot_width=520, plot_height=520,x_axis_type="datetime")
    p2.line(df['Date'], df['2_bed'],color="orange", alpha=0.8)
    p2.title = "2 Beds Listing Price Trend in" +zipcode
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Median Price'
    p2.ygrid.band_fill_color="olive"
    p2.ygrid.band_fill_alpha = 0.1
    p2.title_text_font_style = "italic"
    p2.yaxis[0].formatter = NumeralTickFormatter(format="$0,0.00")
    script2, div2 = components(p2)
    return render_template('trend.html',script1 = script1,script2 = script2,div1 = div1, div2 = div2,zipcode = zipcode)

@app.route('/DataSet',methods = ["GET"])
def dataset():
    return render_template('DataSet.html')

@app.route('/FeaturePlot',methods = ["GET"])
def feature():

    # House Price Vs #Bedroom
    p1 = figure(width=550, height=500,background_fill='#e9e0db')
    p1.circle(ri.df_final['Beds'], ri.df_final['Housing_price'], fill_color="#cc6633",line_color = "#cc6633",size = 20,alpha = 0.5)
    p1.title = "House Price Vs #Bedroom"
    p1.grid.grid_line_alpha=0
    p1.xaxis.axis_label = 'Number of Bedrooms'
    p1.yaxis.axis_label = 'House Price Per SQFT'
    p1.ygrid.band_fill_color="olive"
    p1.ygrid.band_fill_alpha = 0.1
    script1, div1 = components(p1)

    
    defaults.width = 550
    defaults.height = 500
    p2 = Bar(ri.df_final,label = 'Beds',values = 'Housing_price',agg= 'mean')   
    script2, div2 = components(p2)


    # House Price Vs #Bathroom
    p3 = figure(width=550, height=500,background_fill='#e9e0db')
    p3.circle(ri.df_final['Baths'], ri.df_final['Housing_price'], fill_color="#cc6633",line_color = "#cc6633",\
            size = 20,alpha = 0.5)
    p3.title = "House Price Vs #Bedroom"
    p3.grid.grid_line_alpha=0
    p3.xaxis.axis_label = 'Number of Bathrooms'
    p3.yaxis.axis_label = 'House Price Per SQFT'
    p3.ygrid.band_fill_color="olive"
    p3.ygrid.band_fill_alpha = 0.1
    script3, div3 = components(p3)

    p4 = Bar(ri.df_final,label = 'Baths',values = 'Housing_price',agg= 'mean')   
    script4, div4 = components(p4)


    p5 = Bar(ri.df_final,label = 'Beds',values = 'Beds',agg= 'count')   
    script5, div5 = components(p5)


    p6 = Bar(ri.df_final,label = 'Baths',values = 'Baths',agg= 'count')   
    script6, div6 = components(p6)


    return render_template('FeaturePlots.html',script1= script1, div1=div1,script2=script2,div2=div2,\
                            script3=script3,div3=div3,script4=script4,div4=div4,script5=script5,div5=div5,\
                            script6=script6,div6=div6)



@app.route('/RegressionAnalysis',methods = ["GET"])
def analysis():
    n_chosen =[20,25,30,35,40,45,50,55,60,65,70,75,80]
    mape_list = [0.2426,0.2351,0.2416,0.2504,0.3117,0.3069,0.27012,0.3109,0.2906,0.3008,0.3187,0.2915,0.3234]
    p = figure(width=800, height=350)
    p.circle(n_chosen, mape_list, color='navy',size = 15, line_color = "#cc6633",alpha = 0.5)
    p.title = "Mean Absolute Percentage Error Vs The Number of Neighbors"
    p.grid.grid_line_alpha=0
    p.xaxis.axis_label = 'Number of Neighbors'
    p.yaxis.axis_label = 'Mean Absolute Percentage Error'
    p.ygrid.band_fill_color="olive"
    p.ygrid.band_fill_alpha = 0.1
    script, div = components(p)
    return render_template('RegressionAnalysis.html',script=script,div=div)


if __name__=="__main__":
    app.config['TRAP_BAD_REQUEST_ERRORS'] = True 
    #app.run(port=33507,debug=True,host='0.0.0.0')
    #app.run(debug=True)
    #app.run(port=33507)
    #app.run() 
    port = int(os.environ.get('PORT',5000))
    app.run(host = '0.0.0.0',port = port, debug=True)
 