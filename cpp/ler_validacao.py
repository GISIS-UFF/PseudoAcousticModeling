
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pasta = Path("/home/juanmarques/workspace/PseudoAcousticModeling/outputs/validacao")
pasta_figuras = pasta / "figuras"

DX = 10.0
DZ = 10.0
DT = 0.0005

parametros = ["q", "epsilon", "delta"]


def lerBin(arquivo):
    with open(arquivo, "rb") as f:
        dimensoes = np.fromfile(f, dtype=np.int32, count=2)

        if len(dimensoes) != 2:
            raise ValueError(f"Cabecalho incompleto: {arquivo}")

        nz, nx = int(dimensoes[0]), int(dimensoes[1])

        if nz <= 0 or nx <= 0:
            raise ValueError(f"Dimensoes invalidas: {arquivo}")

        dados = np.fromfile(f, dtype=np.float32)

    if dados.size != nz * nx:
        raise ValueError(
            f"Tamanho incorreto em {arquivo}: "
            f"esperado {nz * nx}, encontrado {dados.size}"
        )

    if not np.isfinite(dados).all():
        raise ValueError(f"NaN ou Inf no arquivo: {arquivo}")

    return dados.reshape(nz, nx)


def lerMetricas():
    arquivo = pasta / "metrics.csv"
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo}")

    metricas = {}
    outras_metricas = []

    with open(arquivo, "r", newline="") as f:
        leitor = csv.DictReader(f)

        for linha in leitor:
            teste = linha["test"]
            entrada = int(linha["input"])
            saida = int(linha["output"])
            h = float(linha["h"])

            if teste == "gradient":
                nome = "gradiente_" + parametros[entrada]
            elif teste == "born":
                nome = "born_" + parametros[entrada]
            elif teste == "hessian":
                nome = (
                    "hessiana_" + parametros[entrada] +
                    "_para_" + parametros[saida] +
                    "_h" + f"{h:.5f}"
                )
            else:
                outras_metricas.append(linha)
                continue

            metricas[nome] = linha

    return metricas, outras_metricas


def fazerFigura(nome, operador, fd, metrica):
    diferenca = operador.astype(np.float64) - fd.astype(np.float64)

    limite = max(
        np.max(np.abs(operador)),
        np.max(np.abs(fd)),
        np.max(np.abs(diferenca))
    )
    if limite == 0:
        limite = 1.0

    eh_born = nome.startswith("born_")
    nz, nx = operador.shape

    if eh_born:
        extent = (0, nx - 1, (nz - 1) * DT, 0)
        xlabel = "Receptor"
        ylabel = "Tempo (s)"
        aspecto = "auto"
    else:
        extent = (0, (nx - 1) * DX, (nz - 1) * DZ, 0)
        xlabel = "x (m)"
        ylabel = "z (m)"
        aspecto = "equal"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    titulos = ["Operador implementado", "Diferencas finitas", "Implementado - FD"]

    for ax, dados, titulo in zip(axes, [operador, fd, diferenca], titulos):
        im = ax.imshow(
            dados, cmap="RdBu_r", vmin=-limite, vmax=limite,
            extent=extent, aspect=aspecto, interpolation="nearest"
        )
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.colorbar(im, ax=axes, shrink=0.8, format="%.2e")

    if metrica is not None:
        erro = float(metrica["relative_error"])
        fig.suptitle(f"{nome}  |  Erro relativo = {erro:.3e}")
    else:
        fig.suptitle(nome)

    destino = pasta_figuras / (nome + ".png")
    fig.savefig(destino, dpi=180)
    plt.close(fig)
    return destino


def main():
    metricas, outras_metricas = lerMetricas()
    arquivos = sorted(pasta.glob("*_operator.bin"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum *_operator.bin encontrado em {pasta}")

    pasta_figuras.mkdir(parents=True, exist_ok=True)

    print("\nResultados de validacao\n")
    print(f"{'Teste':55s} {'Erro relativo':>15s}")
    print("-" * 72)

    for arquivo in arquivos:
        nome = arquivo.name.removesuffix("_operator.bin")
        arquivo_fd = pasta / (nome + "_fd.bin")

        if not arquivo_fd.exists():
            raise FileNotFoundError(f"Falta o par: {arquivo_fd}")

        operador = lerBin(arquivo)
        fd = lerBin(arquivo_fd)

        if operador.shape != fd.shape:
            raise ValueError(f"Dimensoes diferentes: {nome}")

        metrica = metricas.get(nome)
        if metrica is not None:
            print(f"{nome:55s} {float(metrica['relative_error']):15.5e}")
        else:
            print(f"{nome:55s} {'sem metrica':>15s}")

        fazerFigura(nome, operador, fd, metrica)

    if outras_metricas:
        print("\nMetricas adicionais (repetibilidade, curvatura, simetria):")
        for linha in outras_metricas:
            print(
                f"{linha['test']:15s} entrada={linha['input']:>2s} "
                f"saida={linha['output']:>2s} "
                f"erro={float(linha['relative_error']):.5e} "
                f"norma_1={float(linha['operator_norm']):.5e} "
                f"norma_2={float(linha['fd_norm']):.5e}"
            )

    print(f"\n{len(arquivos)} figuras geradas em: {pasta_figuras.resolve()}")


if __name__ == "__main__":
    main()

