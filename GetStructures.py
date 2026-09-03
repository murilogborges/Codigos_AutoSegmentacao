import pydicom
from tkinter import Tk, filedialog, Text, Button, END

def copiar_para_clipboard(texto, janela):
    janela.clipboard_clear()
    janela.clipboard_append(texto)
    janela.update()  # Atualiza o clipboard
    print("Nomes copiados para a área de transferência!")

# Criar janela principal oculta para seleção de arquivos
root = Tk()
root.withdraw()

# Abrir janela para selecionar múltiplos arquivos
rtstruct_paths = filedialog.askopenfilenames(
    title="Selecione os arquivos RTSTRUCT DICOM",
    filetypes=[("DICOM files", "*.dcm"), ("Todos os arquivos", "*.*")],
    initialdir="/home"
)

if rtstruct_paths:
    # Criar nova janela para mostrar estruturas
    janela = Tk()
    janela.title("Estruturas dos RTSTRUCTs")

    texto = Text(janela, width=80, height=30)
    texto.pack()

    nomes = []
    ds_base = None

    for idx, rtstruct_path in enumerate(rtstruct_paths):
        ds = pydicom.dcmread(rtstruct_path)
        if ds_base is None:
            ds_base = ds  # Usar o primeiro como base para salvar depois

        texto.insert(END, f"\nArquivo {idx+1}: {rtstruct_path}\n")
        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                roi_number = roi.ROINumber
                roi_name = roi.ROIName
                nomes.append(roi_name)
                texto.insert(END, f"ROI {roi_number}: {roi_name}\n")
        else:
            texto.insert(END, "Nenhuma estrutura encontrada.\n")

    # Botão para copiar nomes
    botao = Button(
        janela,
        text="Copiar nomes para área de transferência",
        command=lambda: copiar_para_clipboard("\n".join(nomes), janela)
    )
    botao.pack()

    # Salvar novo arquivo combinado (ainda apenas o primeiro como base)
    if ds_base is not None:
        ds_base.save_as("GrupoMama.dcm")
        texto.insert(END, "\nNovo arquivo salvo como GrupoMama.dcm\n")

    janela.mainloop()
else:
    print("Nenhum arquivo selecionado.")
