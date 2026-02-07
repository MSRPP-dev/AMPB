#!/bin/bash
# entrypoint.sh

# Instalar AMPBLib
pip install -e /AMPBLib --root-user-action=ignore
	
# Iniciar el demonio cron en segundo plano
# -L 8: Nivel de log (0-8, 8 es verboso)
echo "Starting cron..."
cron -f -L 8 &

# Mantener el contenedor corriendo y mostrar el log de cron en stdout
echo "Tailing /var/log/cron.log..."
tail -f /var/log/cron.log