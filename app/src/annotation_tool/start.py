
import os
import json
import pandas as pd
from wtforms import TextField
from wtforms.validators import DataRequired
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from flask import Flask, escape, request, redirect, url_for, current_app, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import Form
import numpy as np
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = os.path.join("sqlite:///{}".format(os.path.join(project_dir, "data")),"annotations.db")

app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["SQLALCHEMY_DATABASE_URI"] = database_file
app.secret_key = 'xxxxyyyyyzzzzz'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

db = SQLAlchemy(app)
used_template = 'index_autocomplete.html'
DATASET = 'data/examples.xlsx'

json_file = open('structure/annotations_structure.json')
json_structure = str(json.load(json_file))

json_leaves = {}
json_leaves_by_group = {}

class LoginForm(Form):
    """Form class for user login."""
    username = TextField('username', validators=[DataRequired()])

class Annotation(db.Model):
    id = db.Column(db.Integer , primary_key=True)
    identifier = db.Column(db.String(80), unique=False, nullable=False)
    annotation = db.Column(db.String(80), unique=False, nullable=False)
    user = db.Column(db.String(80), unique=False, nullable=False)

    def __repr__(self):
        return "<Identifier: {}>".format(self.identifier)

class User(db.Model):
    __tablename__ = 'user'

    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)
    qc = db.Column(db.Boolean, default=False)

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.email

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False

@login_manager.user_loader
def user_loader(user_id):
    """Given *user_id*, return the associated User object.

    :param unicode user_id: user_id (email) user to retrieve

    """
    return User.query.get(user_id)

def recursive_read(current_array, data_array, leaf_collection):
    if len(data_array[1]) < 1:
        return current_array
    for key in data_array[1].keys():
        #print("Adding "+key)
        current_array.append(str(key))
        added_array = recursive_read([],data_array[1][key], leaf_collection)
        if len(added_array) > 1:
            current_array.append(added_array)
        else:
            leaf_collection[str(key)] = data_array[1][key][0]
    return current_array

myTree = ['DECISION']
with open('structure/annotations_structure.json') as json_file:
    data = json.load(json_file)
    myTree.append(recursive_read([], ["",data], json_leaves))
    for key in data.keys():
        leaves = {}
        recursive_read([], data[key],leaves)
        json_leaves_by_group[key] = list(leaves.keys())

data = pd.read_excel(DATASET, sheet_name = 'Sheet1')
data['EMPI_ENC'] = data['EMPI_ENC'].astype(str).str.replace('\\W', '')
data['ACC_NBR_ENC'] = data['ACC_NBR_ENC'].astype(str).str.replace('\\W', '')

data['identifier'] = data['EMPI_ENC'].map(str)+'.' + data['ACC_NBR_ENC'].map(str) + '.' + data['PART_DESIGNATOR'].map(str)
count_before = len(data.index)
data.set_index('identifier', inplace=True)
duplicates = data.index[data.index.duplicated(keep='first')].tolist()
# Used for debugging
#print(data.loc['10001290.10002445.nan'])
for d in duplicates:
    dupes = data.loc[d]
    if len(np.unique(dupes['EMPI_ENC'].notna()))!=1 or len(np.unique(dupes['ACC_NBR_ENC'].notna()))!=1 or len(np.unique(dupes['PART_DESIGNATOR'].notna()))!=1:
        raise Exception("Duplicate IDs are not the same! For duplicate ID: {}".format(dupes.index.tolist()[0]))
data = data.loc[~data.index.duplicated(keep='first')]
identifiers = data.index.values.tolist()
count_after = len(data.index)
print("Found {}/{} duplicate IDs!".format(count_before-count_after, count_before))
currently_annotated = 10

def find_missing_annotation():
    global currently_annotated
    currently_annotated = 0
    for i in identifiers:

        if not(Annotation.query.filter_by(identifier=i).first()):
            return i
        currently_annotated += 1
    return 'END'

def find_missing_annotation_qc():
    for i in identifiers:
        if not(Annotation.query.filter_by(identifier=i).filter_by(user=current_user.email).first()):
            return i
    return 'END'

def add_annotation(identifier, label,user):
    existing_annotation = Annotation.query.filter_by(identifier=identifier, user=user).first()
    if existing_annotation:
        existing_annotation.annotation = label
        db.session.commit()
    else:
        annotation = Annotation(identifier=identifier, annotation = label, user = user)
        db.session.add(annotation)
        db.session.commit()

def get_annotation(identifier):
    annotation = Annotation.query.filter_by(identifier=identifier).first()
    if(annotation):
        annotation2 = Annotation.query.filter_by(identifier=identifier).filter_by(user=current_user.email).first()
        if(annotation2):
            return annotation2.annotation.split("|")
        else:
            if not(current_user.qc):
                return annotation.annotation.split("|")
    return []

@app.route('/')
@login_required
def default_route():
    return redirect(url_for('pathology_annotator_default'))

@app.route('/pathology')
@login_required
def pathology_annotator_default():
    if current_user.qc:
        identifier = find_missing_annotation_qc()
    else:
        identifier = find_missing_annotation()
    return redirect(url_for('pathology_annotator',identifier = identifier))

@app.route('/pathology/previous/<identifier>')
@login_required
def pathology_annotator_previous(identifier):
    return redirect(url_for('pathology_annotator',identifier = identifiers[max(0,identifiers.index(identifier)-1)]))

@app.route('/pathology/number/<number>')
@login_required
def pathology_annotator_number(number):
    return redirect(url_for('pathology_annotator',identifier = identifiers[int(number)]))

@app.route('/pathology/next/<identifier>')
@login_required
def pathology_annotator_next(identifier):
    print("Finding Next")
    return redirect(url_for('pathology_annotator',identifier = identifiers[min(len(identifiers)-1,identifiers.index(identifier)+1)]))

@app.route('/pathology/<identifier>')
@login_required
def pathology_annotator(identifier):
    global currently_annotated
    if(identifier == 'END'):
        return 'Finished Annotating! {}/{}'.format(currently_annotated, len(data))
    report_id = data.loc[identifier, 'DOC_ID']
    part_desc = data.loc[identifier, 'PART_DESCR']
    part_text = data.loc[identifier, 'PART_FINAL_TEXT']
    part_designator = data.loc[identifier, 'PART_DESIGNATOR']
    return render_template(used_template, json_structure=json_structure, part_designator = part_designator, report_id = report_id, part_text = part_text, \
    identifier = identifier, leaves = json_leaves, leaves_grouped = json_leaves_by_group, \
    identifier_index = identifiers.index(identifier), total_count = len(identifiers), start_annotation = get_annotation(identifier), user=current_user)

@app.route('/pathology/submit/<identifier>/<label>')
@login_required
def pathology_submit(identifier, label):
    label = label.replace('&', ' ').replace('-','|')
    add_annotation(identifier, label, current_user.email)
    return redirect(url_for('pathology_annotator_next',identifier=identifier))

@app.route('/db')
def view_db():
    annotations = Annotation.query.all()
    return render_template('view_db.html', annotations=annotations)

@app.route('/vdb')
def view_db_table():
    annotations = Annotation.query.all()
    table = [["Report"]]
    users = [u.email for u in User.query.all()]
    records = np.zeros((len(identifiers),len(users)))
    annotations = Annotation.query.all()
    for a in annotations:
        records[identifiers.index(a.identifier),users.index(a.user)] = 1
    return render_template("view_db_for_user.html", records=records.tolist(), users=users)
    
@app.route("/login", methods=["GET", "POST"])
def login():
    """For GET requests, display the login form.
    For POSTS, login the current user by processing the form.

    """
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.get(form.username.data)
        if user:
            user.authenticated = True
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            return redirect(url_for("pathology_annotator_default"))
    return render_template("login.html", form=form)

@app.route("/logout", methods=["GET"])
@login_required
def logout():
    """Logout the current user."""
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    return render_template("logout.html")

@app.route("/create_user")
def create_user():
    return render_template("new_user.html")

@app.route("/new_user/<username>")
def new_user(username):
    user = User(email=username, password="DUMMY_PASSWORD")
    db.session.add(user)
    db.session.commit()
    print('User added.')
    return redirect(url_for("login"))

@app.route("/change_qc/<identifier>")
def change_qc(identifier):
    current_user.qc = not(current_user.qc)
    db.session.add(current_user)
    db.session.commit()
    return redirect(url_for("pathology_annotator",identifier=identifier))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443)
    """
        Extra Requirements to install

        sudo apt install python3-six
        sudo apt install python3-xlrd
        http://10.65.183.233

    """
