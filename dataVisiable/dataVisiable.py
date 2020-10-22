import json

from flask import Flask, redirect, render_template, url_for, request, flash
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np

app = Flask(__name__)

con = create_engine("mysql+pymysql://root:123456@localhost/movie?charset=utf8mb4").connect()
print("connect success!!!!")


@app.route("/movie")
def opinion():
    return render_template("dataVisiable.html")


@app.route("/zhuyan", methods=["GET", "POST"])
def sex():
    res = pd.read_csv("../data/gbDisListToDF.csv")

    total = {}
    dis = []
    for i in zip(res["distance"], res["distanceNum"]):
        dis.append(list(i))

    total = {"dis": dis}
    return total


@app.route("/sortedTypeClassify", methods=["POST", "GET"])
def certification():
    res = pd.read_csv("../data/hour_num_df.csv")
    res = np.array(res).tolist()
    cert_res = []
    cert_num = []
    cert_res_dict = {}

    for i in res:
        cert_res.append(i[0])
        cert_num.append(i[1])

    cert_res_dict["cert_res"] = cert_res
    cert_res_dict["cert_num"] = cert_num

    return cert_res_dict


@app.route("/groupByScore", methods=["GET", "POST"])
def hotSpot():
    res = pd.read_csv("../data/gbtimeListToDF.csv")
    res = np.array(res).tolist()
    score = []
    score_num = []
    score_dict = {}

    for i in res:
        score_num.append(i[0])
        score.append(i[1])

    score_dict["score"] = score
    score_dict["score_num"] = score_num

    return score_dict


@app.route("/daoyan", methods=["GET", "POST"])
def tool():

    locList = []
    locDic = {}
    res = pd.read_csv("../data/borderPointsDF.csv")
    for l in zip(res["longitude"], res["latitude"]):
        locList.append(list(l))

    locDic["loc"] = locList
    return locDic


@app.route("/groupByCountry", methods=["GET", "POST"])
def city():
    locList = []
    locDic = {}
    res = pd.read_csv("../data/locDisDF.csv")
    for l in zip(res["longitude"], res["latitude"]):
        locList.append(list(l))

    locDic["loc"] = locList
    return locDic


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)

# print(gender())
