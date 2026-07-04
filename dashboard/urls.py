from django.urls import path

from dashboard.views import liquidity_dashboard

urlpatterns = [
    path("dashboard/liquidity/", liquidity_dashboard, name="liquidity_dashboard"),
]
