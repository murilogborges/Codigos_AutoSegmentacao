import os
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import draw
import pydicom
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Função para interpretar o escore de Agatson
def interpretar_agatson(score):
    if score == 0:
        return "🟢 Nenhum risco detectável", "green"
    elif 1 <= score <= 100:
        return "🟡 Risco leve", "yellow"
    elif 101 <= score <= 400:
        return "🟠 Risco moderado", "orange"
    elif score > 400:
        return "🔴 Risco alto", "red"
    else:
        return "N/A", "white"

def main():
    root = tk.Tk()
    root.withdraw()

    # Selecionar pasta CT
    ct_folder = filedialog.askdirectory(
        title="Selecione a pasta com os arquivos CT (DICOM)",
        initialdir="/home"
    )
    if not ct_folder:
        messagebox.showerror("Erro", "Nenhuma pasta selecionada.")
        root.destroy()
        return

    # Selecionar RTSTRUCT
    parent_folder = Path(ct_folder).parent
    rtstruct_path = filedialog.askopenfilename(
        initialdir="/home",
        title="Selecione o arquivo RTSTRUCT",
        filetypes=[("DICOM files", "*.dcm")]
    )
    if not rtstruct_path:
        messagebox.showerror("Erro", "Nenhum arquivo RTSTRUCT selecionado.")
        root.destroy()
        return

    # Carregar CT
    slices = [pydicom.dcmread(os.path.join(ct_folder, f))
              for f in sorted(os.listdir(ct_folder)) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    ct_image = np.stack([s.pixel_array for s in slices], axis=-1)
    slope = getattr(slices[0], 'RescaleSlope', 1)
    intercept = getattr(slices[0], 'RescaleIntercept', 0)
    ct_image = ct_image * slope + intercept
    pixel_spacing = list(slices[0].PixelSpacing) + [float(slices[0].SliceThickness)]
    origin = slices[0].ImagePositionPatient
    patient_name = getattr(slices[0], "PatientName", "Desconhecido")
    patient_id = getattr(slices[0], "PatientID", "Desconhecido")

    # Carregar RTSTRUCT
    rtstruct = pydicom.dcmread(rtstruct_path)
    structures = {}
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name = roi.ROIName
        for contour in rtstruct.ROIContourSequence:
            if contour.ReferencedROINumber == roi_number:
                structures[roi_name] = contour.ContourSequence
                break

    # Funções auxiliares
    def get_mask_from_contours(contours):
        mask = np.zeros(ct_image.shape, dtype=np.uint8)
        for contour in contours:
            points = np.array(contour.ContourData).reshape(-1, 3)
            x_coords = (points[:, 0] - origin[0]) / pixel_spacing[0]
            y_coords = (points[:, 1] - origin[1]) / pixel_spacing[1]
            z_index = int(round((points[0, 2] - origin[2]) / pixel_spacing[2]))
            if 0 <= z_index < ct_image.shape[2]:
                rr, cc = draw.polygon(y_coords, x_coords, shape=ct_image.shape[:2])
                mask[rr, cc, z_index] = 1
        return mask

    def dfs(matrix, x, y, visited, cluster, threshold=129):
        directions = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        visited[x, y] = True
        cluster.append((x, y))
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if (0 <= nx < matrix.shape[0] and
                0 <= ny < matrix.shape[1] and
                not visited[nx, ny] and
                matrix[nx, ny] > threshold):
                dfs(matrix, nx, ny, visited, cluster, threshold)

    def find_clusters(matrix, threshold=129, min_size=4):
        visited = np.zeros(matrix.shape, dtype=bool)
        clusters = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > threshold and not visited[i, j]:
                    cluster = []
                    dfs(matrix, i, j, visited, cluster, threshold)
                    if len(cluster) >= min_size:
                        clusters.append(cluster)
        return clusters

    def agatson_score(filtered_array, cluster):
        score, volume_score, maximum = 0, 0, 0
        for x, y in cluster:
            if 0 <= x < filtered_array.shape[0] and 0 <= y < filtered_array.shape[1]:
                value = filtered_array[x, y]
                maximum = max(maximum, value)
                score += pixel_spacing[0] * pixel_spacing[1]
                volume_score += pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]
        if 130 <= maximum <= 199:
            final_score = score
        elif 200 <= maximum <= 299:
            final_score = 2 * score
        elif 300 <= maximum <= 399:
            final_score = 3 * score
        elif maximum >= 400:
            final_score = 4 * score
        else:
            final_score = 0
        return round(final_score, 1), round(volume_score, 1)

    # Processar estruturas
    results = []
    for roi_name, contours in structures.items():
        mask = get_mask_from_contours(contours)
        total_score, total_volume = 0, 0
        for i in range(mask.shape[2]):
            masked_slice = np.where(mask[:, :, i] > 0, ct_image[:, :, i], np.nan)
            rows, cols = ~np.all(np.isnan(masked_slice), axis=1), ~np.all(np.isnan(masked_slice), axis=0)
            filtered_array = masked_slice[rows][:, cols]
            clusters = find_clusters(filtered_array, threshold=129)
            for cluster in clusters:
                s, v = agatson_score(filtered_array, cluster)
                total_score += s
                total_volume += v
        risco, cor = interpretar_agatson(total_score)
        results.append({
            'ROI': roi_name,
            'Agatson_score': round(total_score, 1),
            'Volume_score': round(total_volume, 1),
            'Interpretação': risco,
            'Cor': cor
        })

    df = pd.DataFrame(results).sort_values(by="Agatson_score", ascending=False)

    # Mostrar resultados
    win = tk.Tk()
    win.title(f"Resultados - Paciente: {patient_name} (ID: {patient_id})")

    tree = ttk.Treeview(win, columns=("ROI", "Agatson_score", "Volume_score", "Interpretação"), show="headings")
    tree.heading("ROI", text="Estrutura")
    tree.heading("Agatson_score", text="Agatson Score")
    tree.heading("Volume_score", text="Volume Score")
    tree.heading("Interpretação", text="Risco")
    tree.pack(fill="both", expand=True)

    style = ttk.Style()
    style.configure("Treeview", rowheight=25)

    for _, row in df.iterrows():
        item = tree.insert("", "end", values=(row["ROI"], row["Agatson_score"], row["Volume_score"], row["Interpretação"]))
        tree.item(item, tags=(row["Cor"],))

    tree.tag_configure("green", background="lightgreen")
    tree.tag_configure("yellow", background="lightyellow")
    tree.tag_configure("orange", background="orange")
    tree.tag_configure("red", background="tomato")

    def copiar():
        text = df.drop(columns="Cor").to_string(index=False)
        win.clipboard_clear()
        win.clipboard_append(text)
        messagebox.showinfo("Copiado", "Resultados copiados para a área de transferência.")

    btn = tk.Button(win, text="Copiar valores", command=copiar)
    btn.pack(pady=10)

    # Encerrar corretamente ao fechar
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.mainloop()

if __name__ == "__main__":
    main()
