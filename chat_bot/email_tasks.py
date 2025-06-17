from celery import shared_task
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.conf import settings

@shared_task
def send_email_task(pdf_url):
    print("Sending email for PDF:", pdf_url)
    try:
        subject = "📄 Your PDF answer"
        context = {
            "pdf": pdf_url
        }

        # Render HTML email
        html_message = render_to_string('pdf/pdf.html', context)

        send_mail(
            subject="📄 Your PDF answer",
            message="", 
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["dhruvsharma56780@gmail.com"],  
            fail_silently=False,
            html_message=html_message
        )
        print("✅ Email sent successfully.")
    except Exception as e:
        print("Error:", str(e))
