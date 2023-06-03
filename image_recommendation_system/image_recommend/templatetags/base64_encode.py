# base64_encode.py
import base64
from django import template

register = template.Library()

@register.filter
def base64_encode(value):
    return base64.b64encode(value.getvalue()).decode('utf-8')
