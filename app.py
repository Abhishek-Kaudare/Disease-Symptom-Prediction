import numpy as np
from flask import Flask, request, jsonify

import train
import pickle

loaded_model = pickle.load(open('decision_tree_classifier.pickle', 'rb'))

a = train.column_headings


app = Flask(__name__)




@app.route("/symptoms", methods=["POST"])
def syms():
    vect = np.zeros(len(a)-1)
    data = request.get_json(force=True)['data']
    symptoms = [str(s).lower().replace(" ", "_") for s in data]
    print(symptoms)

    for ix in symptoms:
            x = a.index(ix)
            vect[x] = 1


    print(vect)
    desease = loaded_model.predict([vect])[0]

    resp = {
        'desease': desease
        }

    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False, port=5000)
