import graph;
import roundedpath;
unitsize(0.5cm);

real x_range = 4;
real y_scale = 1;

void graph_node (
    pair pos,
    real W, real H,
    real f(real),
    pen color
) {
	filldraw(roundedpath(box(pos, pos+(W, H)), 0.5), color + white, color + linewidth(2));
    path g = graph(f, -x_range, x_range);
    g = scale(1, y_scale) * g;
    pair[] bb = {min(g), max(g)};
    draw(shift(pos) * shift(W/2, H/2) * scale(W / (1.5 * (bb[1].x - bb[0].x))) * shift(0, -(bb[0].y + bb[1].y) / 2) * g, linewidth(1.2));
}

void expr_node (
    pair pos,
    real W, real H,
    string expr,
    pen color
) {
	filldraw(roundedpath(box(pos, pos+(W, H)), 0.5), color + white, color + linewidth(2));
    label(scale(1)*expr, shift(pos) * shift(W/2, H/2) * (0,0));
}

real linear(real x) { return 0.16 * x; }

real scaled_exp(real x) { return 0.08 * exp(x); }

real exp_plus_linear(real x) { return scaled_exp(x) + linear(x); }

real id(real x) { return x; }


real offset = 25;

// ------------------
// Draw arrows
// ------------------

real margin = 0.2;
// Parent to children arrows
draw((2,-margin)--(-2,-3+margin), EndArrow);         // already present
draw((4,-margin)--(8,-3+margin), EndArrow);          // exp_plus_linear → scaled_exp
draw((8,-8-margin)--(8,-11+margin), EndArrow);      // scaled_exp → id
expr_node((2,-3), 2,2, "$+$", blue);
expr_node((5.5,-10.5), 2,2, "$\log$", blue);

draw((2+offset,-margin)--(-2+offset,-3+margin), BeginArrow);         // already present
draw((4+offset,-margin)--(8+offset,-3+margin), BeginArrow);          // exp_plus_linear → scaled_exp
draw((8+offset,-8-margin)--(8+offset,-11+margin), BeginArrow);      // scaled_exp → id
expr_node((2+offset,-3), 2,2, "$+$", blue);
expr_node((5.5+offset,-10.5), 2,2, "$\exp$", blue);

// ------------------
// Draw nodes
// ------------------

graph_node((0,0), 6, 5, exp_plus_linear, red);
graph_node((-5,-8), 6, 5, linear, red);
graph_node((5,-8), 6, 5, scaled_exp, red);
graph_node((5,-16), 6, 5, id, red);

expr_node((offset,0), 6, 5, "$\exp(x) + 0.16x$", RGB(0,200,0));
expr_node((-5+offset,-8), 6, 5, "$0.16x$", RGB(0,200,0));
expr_node((5+offset,-8), 6, 5, "$\exp(x)$", RGB(0,200,0));
expr_node((5+offset,-16), 6, 5, "$x$", RGB(0,200,0));