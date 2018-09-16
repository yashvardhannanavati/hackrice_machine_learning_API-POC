from flask import Flask, request
from firebase_admin import initialize_app
from firebase_admin import credentials
from firebase_admin import db
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True

# Fetch the service account key JSON file contents
cred = credentials.Certificate('ricehack2018-firebase-adminsdk-ukl68-192239081c.json')
# Initialize the app with a service account, granting admin privileges
initialize_app(cred, {
    'databaseURL': 'https://ricehack2018.firebaseio.com/'
})


@app.route('/', methods=['GET'])
def handler():
    discount = request.args.get('discount')
    ref = db.reference('/offers/users/')
    existing_offers = ref.child('729b9a0518abf3f65c9717f2fab9583a').get() or []
    records_df = pd.DataFrame.from_records(existing_offers)
    records_df['offerArrivalDate'] = (pd.to_datetime(
        records_df['offerArrivalDate'].apply(pd.to_numeric), unit='ms'))
    formatted_df = records_df.copy()
    formatted_df.set_index('offerArrivalDate', inplace=True)
    reformatted_df = formatted_df.copy()[['offerPercentage']]
    reformatted_df['offerPercentage'] = (reformatted_df['offerPercentage'].str[:-1].astype(int))
    reformatted_df = reformatted_df.resample("M").apply(lambda x: np.average(x.offerPercentage))
    X = reformatted_df.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    # Test contains values until September 2018, we extend it to predict for October 2018
    predictions = list()
    for t in range(len(test) + 1):
        model = ARIMA(history, order=(1, 0, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)

    if discount:
        if int(discount) >= predictions[-1]:
            return 'Our models suggest you to ACT NOW as this might be the best deal which you would get in the upcoming month(s).'
        else:
            return 'We suggest you to wait a bit longer since the deals look a little bleak. You might get better deals in October.'
    return 'No discount provided!'


if __name__ == '__main__':
    app.run()
