Expedia Hotel Recommandations
==============================

This was one of [Kaggle](http://www.kaggle.com/)'s competition;
[Expedia Hotel Recommandations](https://www.kaggle.com/c/expedia-hotel-recommendations)

It's written for Python 2.7.2 and it's based on ['scikit-learn'](http://scikit-learn.org/)

Project Description
---------------------
It was made to improve recommendation algorithm of Expedia hotel recommendation system.
Expedia wanted to predict which hotel cluster user would belong to.

'**Ensemble_hotel_cluster.py**' is the result to solve the problem.

It was very difficult to find proper model and columns combinations.
However, train dataset has some specific regularities between them;

- user_location_country: The ID of the country the customer is located
- hotel_country: Hotel country
- srch_destination_id: ID of the destination where the hotel search was performed
- nights: How long they hope to stay in hotel
- prepare: How much time they are prepared beforehand

Finally, the target was changed 'hotel cluster' to 'is_package'.

The code file is '**DTC_find_package.py**'

The train file, however, is too big to upload github.
I leave a link where you can download the train file.
There are four files on the web site and only one of them used.

- train.csv

link : https://www.kaggle.com/c/expedia-hotel-recommendations/data
