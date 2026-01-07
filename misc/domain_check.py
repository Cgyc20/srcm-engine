from srcm_engine import Domain, ConversionParams

dom = Domain(length=10.0, n_ssa=100, pde_multiple=5, boundary="zero-flux")
conv = ConversionParams(threshold=100, rate=10.0)

print(dom.h, dom.dx, dom.n_pde)
print(dom.starts[:3], dom.ends[:3])
print(conv)
