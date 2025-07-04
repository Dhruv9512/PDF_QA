from celery import shared_task
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.conf import settings


@shared_task
def send_email_task(data):
   
    try:
        pdf_url = data.get("pdf_url")
        email = data.get("email")
        subject = "📄 Your PDF answer"
        context = {"pdf": pdf_url}
        html_message = render_to_string('pdf/pdf.html', context)

        result = send_mail(
            subject=subject,
            message="",
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[email],
            fail_silently=False,
            html_message=html_message
        )

        print("✅ Email sent successfully. send_mail returned:", result)
        return {"status": "success", "result": result}  # 🔧 Must return something!
    except Exception as e:
        print("❌ Email sending failed:", str(e))
        return {"status": "failed", "error": str(e)}  # 🔧 Must return something!

