# import torch

# Ruta al archivo original (pesado)
ruta_checkpoint = "./output/dfine_hgnetv2_n_custom/best_stg1.pth"

# Ruta donde querés guardar el modelo limpio
ruta_modelo_limpio = "./output/dfine_hgnetv2_n_custom/solo_modelo.pth"

# # Cargar el archivo .pth
# checkpoint = torch.load(ruta_checkpoint, map_location="cpu")

# # Detectar si es un dict con múltiples componentes
# if isinstance(checkpoint, dict):
    # if "model" in checkpoint:
        # print("[✓] 'model' encontrado en el checkpoint.")
        # state_dict = checkpoint["model"]
    # else:
        # print("[!] El checkpoint parece ser un diccionario pero no tiene 'model'. Guardando todo el dict.")
        # state_dict = checkpoint
# else:
    # print("[✓] El archivo es directamente el state_dict del modelo.")
    # state_dict = checkpoint

# # Guardar solo los pesos del modelo
# torch.save(state_dict, ruta_modelo_limpio)

# print(f"\nModelo limpio guardado en: {ruta_modelo_limpio}")


import torch

# Ruta a tu .pth
checkpoint = torch.load(ruta_checkpoint, map_location="cpu")

# Inspeccionar tipo y claves si es un dict
if isinstance(checkpoint, dict):
    print("Es un dict. Claves disponibles:")
    for k in checkpoint.keys():
        print("  -", k)
else:
    print("No es un dict. Es probablemente un state_dict puro.")
    print("Tipo:", type(checkpoint))

# Extraer solo los pesos del modelo
state_dict = checkpoint["model"]

# Guardar un nuevo dict SOLO con el modelo
nuevo_checkpoint = {
    "model": state_dict
}

torch.save(nuevo_checkpoint, ruta_modelo_limpio)

print(f"[✓] Modelo limpio guardado con clave 'model' en: {ruta_modelo_limpio}")