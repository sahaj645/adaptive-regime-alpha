"""Post-signal overlays ('levers') that sit on top of a base regime position.

Each lever is a pure transform from a base position series (+ market data) to a
new position series, kept separate from the regime models so the binary mandate
result stays intact and every lever's marginal effect is measurable in isolation.
"""
