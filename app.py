from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import cv2
import numpy as np
from datetime import datetime

# Define the Flask application
class MobileCoverApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # File storage configuration
        self.UPLOAD_FOLDER = 'uploads'
        self.COVER_TEMPLATES_FOLDER = 'cover_templates'
        self.GENERATED_COVERS_FOLDER = 'generated_covers'
        self._create_folders()

        # SQLite database configuration
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///models.db'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        # Initialize database and migration tools
        self.db = SQLAlchemy(self.app)
        self.migrate = Migrate(self.app, self.db)

        self._define_database_model()

        # Error handler
        self.app.register_error_handler(Exception, self.handle_exception)

        # Initialize routes
        self._initialize_routes()

        # Create the database tables
        with self.app.app_context():
            self.db.create_all()

    def _create_folders(self):
        """Ensure the necessary folders exist."""
        for folder in [self.UPLOAD_FOLDER, self.COVER_TEMPLATES_FOLDER, self.GENERATED_COVERS_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _define_database_model(self):
        """Define the MobileModel table."""
        class MobileModel(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True)
            model_name = self.db.Column(self.db.String(50), unique=True, nullable=False)
            template_filename = self.db.Column(self.db.String(100), nullable=False)
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)

            def __repr__(self):
                return f'<MobileModel {self.model_name}>'

        self.MobileModel = MobileModel

    def handle_exception(self, e):
        """Global error handler."""
        response = {
            "error": str(e),
        }
        return jsonify(response), 500

    def _initialize_routes(self):
        """Initialize the routes for the application."""
        self.app.add_url_rule('/upload', 'upload_image', self.upload_image, methods=['POST'])
        self.app.add_url_rule('/models', 'get_models', self.get_models, methods=['GET'])
        self.app.add_url_rule('/generated_covers', 'get_generated_covers', self.get_generated_covers, methods=['GET'])
        self.app.add_url_rule('/generated_covers/<filename>', 'get_generated_cover', self.get_generated_cover, methods=['GET'])
        self.app.add_url_rule('/add_model', 'add_model', self.add_model, methods=['POST'])
        self.app.add_url_rule('/update_model/<int:model_id>', 'update_model', self.update_model, methods=['PUT'])
        self.app.add_url_rule('/delete_model/<int:model_id>', 'delete_model', self.delete_model, methods=['DELETE'])

    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded!'}), 400

        image = request.files['image']
        model_name = request.form.get('model')

        mobile_model = self.MobileModel.query.filter_by(model_name=model_name).first()
        if not mobile_model:
            return jsonify({'error': 'Invalid mobile model!'}), 400

        if image.filename == '':
            return jsonify({'error': 'No selected file!'}), 400

        # Process the image
        return self._process_image(image, mobile_model)

    def _process_image(self, image, mobile_model):
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        generated_cover_filename = f"{mobile_model.model_name}_{timestamp}.png"
        generated_cover_path = os.path.join(self.GENERATED_COVERS_FOLDER, generated_cover_filename)

        # Save the uploaded image
        filename = secure_filename(image.filename)
        filepath = os.path.join(self.UPLOAD_FOLDER, filename)
        image.save(filepath)

        template_path = os.path.join(self.COVER_TEMPLATES_FOLDER, mobile_model.template_filename)
        if not os.path.exists(template_path):
            return jsonify({'error': 'Mobile cover template not found!'}), 500

        try:
            # Combine the user's image with the cover template
            combined_image = self._combine_images(filepath, template_path)
            cv2.imwrite(generated_cover_path, combined_image)
        except Exception as e:
            return jsonify({'error': f'Error processing images: {str(e)}'}), 500

        return send_file(generated_cover_path, mimetype='image/png')

    def _combine_images(self, user_image_path, template_path):
        cover_template = cv2.imread(template_path)
        user_image = cv2.imread(user_image_path)

        # Resize user image to match cover template dimensions
        user_image = cv2.resize(user_image, (cover_template.shape[1], cover_template.shape[0]))

        # Convert the cover template to HSV and create a mask to remove the green background
        hsv = cv2.cvtColor(cover_template, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        # Remove the green area from the cover template and insert the user image in its place
        cover_without_green = cv2.bitwise_and(cover_template, cover_template, mask=mask_inv)
        user_image_with_green = cv2.bitwise_and(user_image, user_image, mask=mask)

        # Combine the user image with the cover template
        combined_image = cv2.add(cover_without_green, user_image_with_green)

        return combined_image

    def get_models(self):
        models = self.MobileModel.query.all()
        models_data = [
            {
                'id': model.id,
                'model_name': model.model_name,
                'template_filename': model.template_filename,
                'created_at': model.created_at.isoformat()
            }
            for model in models
        ]
        return jsonify(models_data)

    def get_generated_covers(self):
        try:
            files = os.listdir(self.GENERATED_COVERS_FOLDER)
            file_urls = [f"http://192.168.1.27:5000/generated_covers/{file}" for file in files if file.endswith('.png')]
            return jsonify(file_urls), 200
        except Exception as e:
            return jsonify({'error': f'Error retrieving covers: {str(e)}'}), 500

    def get_generated_cover(self, filename):
        try:
            return send_file(os.path.join(self.GENERATED_COVERS_FOLDER, filename), mimetype='image/png')
        except Exception as e:
            return jsonify({'error': f'Error retrieving cover: {str(e)}'}), 500

    def add_model(self):
        if 'template_file' not in request.files or 'model_name' not in request.form:
            return jsonify({'error': 'Model name and template file are required'}), 400

        model_name = request.form.get('model_name')
        template_file = request.files['template_file']

        # Check if the model name already exists
        existing_model = self.MobileModel.query.filter_by(model_name=model_name).first()
        if existing_model:
            return jsonify({'error': 'Model already exists'}), 400

        # Save the new model template
        filename = secure_filename(template_file.filename)
        template_filepath = os.path.join(self.COVER_TEMPLATES_FOLDER, filename)
        template_file.save(template_filepath)

        # Save the model in the database
        new_model = self.MobileModel(model_name=model_name, template_filename=filename)
        self.db.session.add(new_model)
        self.db.session.commit()

        return jsonify({'message': f'Model {model_name} added successfully!'}), 201

    def update_model(self, model_id):
        data = request.get_json()
        model_name = data.get('model_name')
        template_filename = data.get('template_filename')

        model = self.MobileModel.query.get(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404

        if model_name:
            existing_model = self.MobileModel.query.filter_by(model_name=model_name).first()
            if existing_model and existing_model.id != model_id:
                return jsonify({'error': 'Model name already exists'}), 400
            model.model_name = model_name

        if template_filename:
            model.template_filename = template_filename

        self.db.session.commit()

        return jsonify({'message': f'Model {model.model_name} updated successfully!'}), 200

    def delete_model(self, model_id):
        model = self.MobileModel.query.get(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404

        self.db.session.delete(model)
        self.db.session.commit()

        return jsonify({'message': f'Model {model.model_name} deleted successfully!'}), 200


# Expose the app globally so that Flask can detect it
mobile_cover_app = MobileCoverApp()
app = mobile_cover_app.app

# Run the app if this file is executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
