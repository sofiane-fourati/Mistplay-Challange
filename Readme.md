# Mistplay Challange


Requirments for the Notebook:
- Pandas
- sklearn
- xgboost
- Numpy
- Matplotlib
- Pickle

- Django (to open themain interface)

```python
python manage.py runserver 8080
```
This will acces to the port 8080 in the localhost. Then open new tab in the browser: http://localhost:8080/home/

![Home page](https://github.com/sofiane-fourati/Mistplay-Challange/blob/master/Home.png)

Then choose your file and click submit, this will return the result in the next page and save the result in a file in the directory 'predict'

![Results](https://github.com/sofiane-fourati/Mistplay-Challange/blob/master/results.png)

Some mistakes:
- I did not think about other categorical values that may be present in the test set like the nationalities and Android Version which surely will generate an error if trying to predict something.

Solutions:
- Removing those categorical variables (that I can't handle)
OR
- Training each time the model with the new columns

Improvements:
- Data Engineering and adding some other important column like the mark of each phone in the market, year of release, correcting the brand column.
