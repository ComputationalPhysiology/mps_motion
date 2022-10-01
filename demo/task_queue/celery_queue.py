"""
https://flask.palletsprojects.com/en/1.0.x/patterns/celery/

1. Start redis
    redis-server

2. Start celery
    celery -A celery_queue.celery  worker --loglevel=info
    or
    celery -A celery_queue.celery  worker --loglevel=info -P threads

3. Start flask app
    python celery_queue.py

4. Go to localhost:5000/add-test-data

5. Go to localhost:5000/get-displacement/1
    Look at the celery worker

6. Go to localhost:5000/data/1
    Look at the data


"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import mps
import numpy as np
from celery_factory import make_celery
from flask import Flask
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

from mps_motion import Mechanics
from mps_motion import motion_tracking as mt

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL="redis://localhost:6379",
    CELERY_BACKEND="db+sqlite:///celery_mydatabase.db",
    SQLALCHEMY_DATABASE_URI="sqlite:///sql_database.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
celery = make_celery(app)
db = SQLAlchemy(app)

# Folder to store the motion tracking results
# This is numpy data and is very slow to serialize
# so we just save it to a binary file and keep the
# path to the file in the database
MT_FOLDER = Path(__file__).parent.joinpath(".mt_folder")
MT_FOLDER.mkdir(exist_ok=True)

# You should set up a cron job that monitors this folders
# and delete the oldest files whenver the size exceeds
# some limit


@dataclass
class MPSModel(db.Model):  # type: ignore
    id: int = Column(Integer, primary_key=True)
    path: str = Column(String)
    data: Dict[str, str] = Column(JSON, nullable=True)


@celery.task(name="celery_queue.get_mps_displacments")
def get_mps_displacements(data_id, flow_algorithm="farneback", **options):

    db_data = db.session.query(MPSModel).filter(MPSModel.id == data_id).first()

    if db_data is None:
        raise LookupError(f"Could not find data with id {data_id}")

    data = mps.MPS(db_data.path)
    if flow_algorithm in mt.DENSE_FLOW_ALGORITHMS:
        motion = mt.DenseOpticalFlow(data, flow_algorithm=flow_algorithm, **options)
    elif flow_algorithm in mt.SPARSE_FLOW_ALGORITHMS:
        motion = mt.DenseOpticalFlow(data, flow_algorithm=flow_algorithm, **options)
    displacement = motion.get_displacements(scale=0.5)

    path = MT_FOLDER.joinpath(f"displacement_{data_id}.npy").absolute().as_posix()
    np.save(path, displacement)
    print("Done getting displacements")
    db_data.data = {"displacement": path}
    print("Serialize")
    db.session.add(db_data)
    print("Commit")
    db.session.commit()
    print("Done")


@celery.task(name="celery_queue.get_mps_principal_strain")
def get_mps_principal_strain(data_id):

    db_data = db.session.query(MPSModel).filter(MPSModel.id == data_id).first()

    if db_data is None:
        raise LookupError(f"Could not find data with id {data_id}")
    data = db_data.data
    if data is None:
        raise LookupError(f"Could not find data with id {data_id}")
    u_path = data.get("displacement")
    if u_path is None:
        raise LookupError(f"Could not find data with id {data_id}")
    u = np.load(u_path)

    m = Mechanics(u)
    e1 = m.principal_strain()
    print("Done getting principal strain")

    path = MT_FOLDER.joinpath(f"principal_strain_{data_id}.npy").absolute().as_posix()
    np.save(path, e1)

    data["principal_strain"] = path
    db_data.data = data
    print("Serialize")
    print("Commit")
    db.session.commit()
    print("Done")


@app.route("/get-principal-strain/<id>")
def get_principal_strain(id):
    get_mps_principal_strain.delay(id)
    return f"Compute principal strain for data with id {id}!"


@app.route(
    "/get-displacement/<id>",
)
def get_displacements(id):
    get_mps_displacements.delay(id)
    return f"Compute displacement for data with id {id}!"


@app.route("/data/<id>")
def data(id):
    query = db.session.query(MPSModel).filter(MPSModel.id == id)
    if query.count() >= 1:
        data = query.first()
        msg = jsonify(data)
    else:
        msg = f"Could not find any data with id {id}"
    return msg


@app.route("/add-test-data/")
def add_test_data():
    path = Path("../PointH4A_ChannelBF_VC_Seq0018.nd2").absolute().as_posix()
    data = MPSModel(path=path)
    db.session.add(data)
    db.session.commit()
    return f"Add test data {path} with id {data.id}"


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
