"""
Reporting Routes

Provides endpoints for report generation, scheduling, delivery, and management
operations in the AI Automation Bot.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, g, send_file
import os

from ...core.exceptions import ValidationError, ReportingError
from ...core.logger import get_logger

logger = get_logger(__name__)

# Create reporting blueprint
reporting_bp = Blueprint('reporting', __name__)

# In-memory reporting storage (replace with database in production)
reports_db = {}
report_templates_db = {}
report_schedules_db = {}


@reporting_bp.route('/templates', methods=['POST'])
def create_report_template():
    """
    Create a new report template.
    
    Request body:
        - name: Template name
        - description: Template description
        - template_type: Type of template ('html', 'pdf', 'excel', 'json')
        - content: Template content
        - parameters: Template parameters configuration
    
    Returns:
        - 201: Template created successfully
        - 400: Validation error
        - 500: Creation error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        name = data.get('name')
        description = data.get('description', '')
        template_type = data.get('template_type')
        content = data.get('content')
        parameters = data.get('parameters', {})
        
        # Validate required fields
        if not name or not template_type or not content:
            raise ValidationError("name, template_type, and content are required")
        
        # Validate template type
        if template_type not in ['html', 'pdf', 'excel', 'json']:
            raise ValidationError("Invalid template type")
        
        # Create template
        template_id = str(uuid.uuid4())
        template = {
            'id': template_id,
            'user_id': g.user_id,
            'name': name,
            'description': description,
            'template_type': template_type,
            'content': content,
            'parameters': parameters,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'is_active': True
        }
        
        report_templates_db[template_id] = template
        
        logger.info(f"Report template created: {name}")
        
        return jsonify({
            'message': 'Template created successfully',
            'template_id': template_id,
            'template': template
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Create template error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to create template'}), 500


@reporting_bp.route('/templates', methods=['GET'])
def list_templates():
    """
    List all report templates for the current user.
    
    Query parameters:
        - limit: Maximum number of templates to return
        - offset: Number of templates to skip
        - template_type: Filter by template type
    
    Returns:
        - 200: List of templates
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        template_type = request.args.get('template_type')
        
        # Filter templates by user
        user_templates = [
            template for template in report_templates_db.values()
            if template['user_id'] == g.user_id
        ]
        
        # Apply filters
        if template_type:
            user_templates = [t for t in user_templates if t['template_type'] == template_type]
        
        # Sort by creation date (newest first)
        user_templates.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_templates)
        templates = user_templates[offset:offset + limit]
        
        return jsonify({
            'templates': templates,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List templates error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list templates'}), 500


@reporting_bp.route('/templates/<template_id>', methods=['GET'])
def get_template(template_id: str):
    """
    Get specific report template.
    
    Args:
        template_id: Template identifier
    
    Returns:
        - 200: Template details
        - 404: Template not found
        - 403: Access denied
    """
    try:
        template = report_templates_db.get(template_id)
        
        if not template:
            return jsonify({'error': 'Not Found', 'message': 'Template not found'}), 404
        
        # Check access permission
        if template['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'template': template
        }), 200
        
    except Exception as e:
        logger.error(f"Get template error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get template'}), 500


@reporting_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Generate a report using a template.
    
    Request body:
        - template_id: Template identifier
        - parameters: Report parameters
        - output_format: Output format ('html', 'pdf', 'excel', 'json')
        - report_name: Name for the report
    
    Returns:
        - 200: Report generated successfully
        - 400: Validation error
        - 404: Template not found
        - 403: Access denied
        - 500: Generation error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        template_id = data.get('template_id')
        parameters = data.get('parameters', {})
        output_format = data.get('output_format')
        report_name = data.get('report_name')
        
        # Validate required fields
        if not template_id or not output_format or not report_name:
            raise ValidationError("template_id, output_format, and report_name are required")
        
        # Get template
        template = report_templates_db.get(template_id)
        
        if not template:
            return jsonify({'error': 'Not Found', 'message': 'Template not found'}), 404
        
        # Check access permission
        if template['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Generate report
        report_id = str(uuid.uuid4())
        report_data = _generate_report_content(template, parameters, output_format)
        
        # Create report record
        report = {
            'id': report_id,
            'user_id': g.user_id,
            'template_id': template_id,
            'name': report_name,
            'output_format': output_format,
            'parameters': parameters,
            'file_path': report_data['file_path'],
            'file_size': report_data['file_size'],
            'generated_at': datetime.utcnow(),
            'download_count': 0
        }
        
        reports_db[report_id] = report
        
        logger.info(f"Report generated: {report_name}")
        
        return jsonify({
            'message': 'Report generated successfully',
            'report_id': report_id,
            'report': report
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Generate report error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to generate report'}), 500


def _generate_report_content(template: Dict[str, Any], parameters: Dict[str, Any], 
                           output_format: str) -> Dict[str, Any]:
    """Generate report content based on template and parameters."""
    try:
        template_type = template['template_type']
        content = template['content']
        
        # Create reports directory
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{template['name']}_{timestamp}.{output_format}"
        file_path = os.path.join(reports_dir, filename)
        
        # Generate content based on format
        if output_format == 'html':
            html_content = _render_html_template(content, parameters)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        elif output_format == 'json':
            json_content = _render_json_template(content, parameters)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, default=str)
        
        elif output_format == 'pdf':
            # This would integrate with a PDF generation library
            pdf_content = _render_pdf_template(content, parameters)
            with open(file_path, 'wb') as f:
                f.write(pdf_content)
        
        elif output_format == 'excel':
            # This would integrate with an Excel generation library
            excel_content = _render_excel_template(content, parameters)
            with open(file_path, 'wb') as f:
                f.write(excel_content)
        
        else:
            raise ReportingError(f"Unsupported output format: {output_format}")
        
        return {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path)
        }
        
    except Exception as e:
        logger.error(f"Report content generation error: {e}")
        raise ReportingError(f"Failed to generate report content: {str(e)}")


def _render_html_template(content: str, parameters: Dict[str, Any]) -> str:
    """Render HTML template with parameters."""
    try:
        # Simple template rendering (replace with proper templating engine)
        rendered_content = content
        
        for key, value in parameters.items():
            placeholder = f"{{{{{key}}}}}"
            rendered_content = rendered_content.replace(placeholder, str(value))
        
        return rendered_content
        
    except Exception as e:
        logger.error(f"HTML template rendering error: {e}")
        raise ReportingError(f"Failed to render HTML template: {str(e)}")


def _render_json_template(content: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Render JSON template with parameters."""
    try:
        # Parse content as JSON template
        template_data = json.loads(content)
        
        # Replace parameters in template
        def replace_params(obj):
            if isinstance(obj, dict):
                return {k: replace_params(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_params(item) for item in obj]
            elif isinstance(obj, str):
                for key, value in parameters.items():
                    placeholder = f"{{{{{key}}}}}"
                    obj = obj.replace(placeholder, str(value))
                return obj
            else:
                return obj
        
        return replace_params(template_data)
        
    except Exception as e:
        logger.error(f"JSON template rendering error: {e}")
        raise ReportingError(f"Failed to render JSON template: {str(e)}")


def _render_pdf_template(content: str, parameters: Dict[str, Any]) -> bytes:
    """Render PDF template with parameters."""
    try:
        # This would integrate with a PDF generation library like reportlab
        # For now, return a simple PDF content
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
    except Exception as e:
        logger.error(f"PDF template rendering error: {e}")
        raise ReportingError(f"Failed to render PDF template: {str(e)}")


def _render_excel_template(content: str, parameters: Dict[str, Any]) -> bytes:
    """Render Excel template with parameters."""
    try:
        # This would integrate with an Excel generation library like openpyxl
        # For now, return a simple Excel content
        return b"PK\x03\x04\x14\x00\x00\x00\x08\x00"
        
    except Exception as e:
        logger.error(f"Excel template rendering error: {e}")
        raise ReportingError(f"Failed to render Excel template: {str(e)}")


@reporting_bp.route('/reports', methods=['GET'])
def list_reports():
    """
    List all generated reports for the current user.
    
    Query parameters:
        - limit: Maximum number of reports to return
        - offset: Number of reports to skip
        - output_format: Filter by output format
    
    Returns:
        - 200: List of reports
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        output_format = request.args.get('output_format')
        
        # Filter reports by user
        user_reports = [
            report for report in reports_db.values()
            if report['user_id'] == g.user_id
        ]
        
        # Apply filters
        if output_format:
            user_reports = [r for r in user_reports if r['output_format'] == output_format]
        
        # Sort by generation date (newest first)
        user_reports.sort(key=lambda x: x['generated_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_reports)
        reports = user_reports[offset:offset + limit]
        
        return jsonify({
            'reports': reports,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List reports error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list reports'}), 500


@reporting_bp.route('/reports/<report_id>', methods=['GET'])
def get_report(report_id: str):
    """
    Get specific report.
    
    Args:
        report_id: Report identifier
    
    Returns:
        - 200: Report details
        - 404: Report not found
        - 403: Access denied
    """
    try:
        report = reports_db.get(report_id)
        
        if not report:
            return jsonify({'error': 'Not Found', 'message': 'Report not found'}), 404
        
        # Check access permission
        if report['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'report': report
        }), 200
        
    except Exception as e:
        logger.error(f"Get report error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get report'}), 500


@reporting_bp.route('/reports/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    """
    Download report file.
    
    Args:
        report_id: Report identifier
    
    Returns:
        - 200: Report file
        - 404: Report not found
        - 403: Access denied
        - 500: Download error
    """
    try:
        report = reports_db.get(report_id)
        
        if not report:
            return jsonify({'error': 'Not Found', 'message': 'Report not found'}), 404
        
        # Check access permission
        if report['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Check if file exists
        if not os.path.exists(report['file_path']):
            return jsonify({'error': 'Not Found', 'message': 'Report file not found'}), 404
        
        # Update download count
        report['download_count'] += 1
        
        # Send file
        return send_file(
            report['file_path'],
            as_attachment=True,
            download_name=os.path.basename(report['file_path'])
        )
        
    except Exception as e:
        logger.error(f"Download report error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to download report'}), 500


@reporting_bp.route('/schedules', methods=['POST'])
def create_report_schedule():
    """
    Create a new report schedule.
    
    Request body:
        - name: Schedule name
        - template_id: Template identifier
        - schedule: Schedule configuration
        - parameters: Report parameters
        - output_format: Output format
        - recipients: List of recipient emails
    
    Returns:
        - 201: Schedule created successfully
        - 400: Validation error
        - 500: Creation error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        name = data.get('name')
        template_id = data.get('template_id')
        schedule = data.get('schedule')
        parameters = data.get('parameters', {})
        output_format = data.get('output_format')
        recipients = data.get('recipients', [])
        
        # Validate required fields
        if not name or not template_id or not schedule or not output_format:
            raise ValidationError("name, template_id, schedule, and output_format are required")
        
        # Validate template exists
        template = report_templates_db.get(template_id)
        if not template:
            raise ValidationError("Template not found")
        
        # Create schedule
        schedule_id = str(uuid.uuid4())
        report_schedule = {
            'id': schedule_id,
            'user_id': g.user_id,
            'name': name,
            'template_id': template_id,
            'schedule': schedule,
            'parameters': parameters,
            'output_format': output_format,
            'recipients': recipients,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'is_active': True,
            'last_generated': None,
            'next_generation': _calculate_next_generation(schedule)
        }
        
        report_schedules_db[schedule_id] = report_schedule
        
        logger.info(f"Report schedule created: {name}")
        
        return jsonify({
            'message': 'Schedule created successfully',
            'schedule_id': schedule_id,
            'schedule': report_schedule
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Create schedule error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to create schedule'}), 500


def _calculate_next_generation(schedule: Dict[str, Any]) -> Optional[datetime]:
    """Calculate next report generation time based on schedule."""
    try:
        schedule_type = schedule.get('type')
        now = datetime.utcnow()
        
        if schedule_type == 'daily':
            time_str = schedule.get('time', '00:00')
            hour, minute = map(int, time_str.split(':'))
            next_gen = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_gen <= now:
                next_gen += timedelta(days=1)
            
            return next_gen
        
        elif schedule_type == 'weekly':
            day = schedule.get('day', 'monday')
            time_str = schedule.get('time', '00:00')
            hour, minute = map(int, time_str.split(':'))
            
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            target_day = day_map.get(day.lower(), 0)
            current_day = now.weekday()
            
            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7
            
            next_gen = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            next_gen += timedelta(days=days_ahead)
            
            return next_gen
        
        else:
            return None
            
    except Exception:
        return None


@reporting_bp.route('/schedules', methods=['GET'])
def list_schedules():
    """
    List all report schedules for the current user.
    
    Query parameters:
        - limit: Maximum number of schedules to return
        - offset: Number of schedules to skip
    
    Returns:
        - 200: List of schedules
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Filter schedules by user
        user_schedules = [
            schedule for schedule in report_schedules_db.values()
            if schedule['user_id'] == g.user_id
        ]
        
        # Sort by creation date (newest first)
        user_schedules.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_schedules)
        schedules = user_schedules[offset:offset + limit]
        
        return jsonify({
            'schedules': schedules,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List schedules error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list schedules'}), 500 