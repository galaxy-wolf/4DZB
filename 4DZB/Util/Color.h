#pragma once

struct Color3f {
	Color3f(float R, float G, float B) :r(R), g(G), b(B) {}
	Color3f(float white) :r(white), g(white), b(white) {}
	Color3f() = default;

	float r, g, b;
};
