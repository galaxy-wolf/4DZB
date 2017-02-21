#pragma once
#include <glm/glm.hpp>
struct AreaLightDes
{
	// basic variables
	float  rightRadius;	//half width
	float  topRadius;	//half height
	glm::vec3 position;	//light position
	glm::vec3 viewDir;		//light view direction(normalized)
	glm::vec3 upDir;		//up direction(normalized)

	int lightResWidth;
	int lightResHeight;
};