(base) kibira@DESKTOP-EUV20F0:~/fire-prediction-system/model_training$ curl -X POST http://127.0.0.1:5000/predict \
>      -H "Content-Type: application/json" \
>      -d '{
>            "sensor_timestamp": "2025-04-09T12:00:00",
>            "modis_timestamp": "2025-04-09T12:30:00",
>            "sensor_value": 0.8,
>            "fire_lat": 12.345,
>            "fire_long": 67.890,
>            "bright_ti4": 300,
>            "confidence": "h",
>            "fire_radiative_power": 5000,
>            "daynight": "D"
>          }'
{
  "prediction": 1
}
(base) kibira@DESKTOP-EUV20F0:~/fire-prediction-system/model_training$ 