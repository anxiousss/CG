#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    vec3 ambient_light_intensity;
    vec3 sun_light_direction;
    vec3 sun_light_color;

    uint point_light_count;
    uint spot_light_count;
};

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    vec3 specular_color;
    float shininess;
};

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle;
    vec3 color;
    float _pad0;
};

layout(binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(binding = 3, std430) readonly buffer SpotLights {  // binding = 3
    SpotLight spot_lights[];
};

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    // Ambient lighting
    vec3 ambient = ambient_light_intensity * albedo_color;

    // Sun lighting (directional light)
    vec3 sun_light_dir = normalize(sun_light_direction);
    float sun_diffuse = max(dot(normal, -sun_light_dir), 0.0);
    vec3 sun_diffuse_color = sun_diffuse * sun_light_color * albedo_color;

    // Sun specular (Blinn-Phong)
    vec3 sun_half_dir = normalize(-sun_light_dir + view_dir);
    float sun_specular = pow(max(dot(normal, sun_half_dir), 0.0), shininess);
    vec3 sun_specular_color = sun_specular * sun_light_color * specular_color;

    vec3 total_light = ambient + sun_diffuse_color + sun_specular_color;

    // Point lights
    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];
        vec3 light_dir = light.position - f_position;
        float distance = length(light_dir);
        light_dir = normalize(light_dir);

        // Check if within radius
        if (distance > light.radius) {
            continue;
        }

        // Attenuation (quadratic falloff)
        float attenuation = 1.0 - (distance / light.radius);
        attenuation *= attenuation;

        // Diffuse
        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo_color;

        // Specular (Blinn-Phong)
        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_color = specular * light.color * specular_color;

        total_light += (diffuse_color + specular_color) * attenuation;
    }

    // Spot lights
    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        vec3 light_dir = light.position - f_position;
        float distance = length(light_dir);
        light_dir = normalize(light_dir);

        // Check if within radius
        if (distance > light.radius) {
            continue;
        }

        // Spot light cone check
        float cos_theta = dot(light_dir, normalize(-light.direction));
        if (cos_theta < light.angle) {
            continue;
        }

        // Attenuation (quadratic falloff)
        float attenuation = 1.0 - (distance / light.radius);
        attenuation *= attenuation;

        // Soft edge for spot light
        float spot_factor = 1.0;
        if (light.angle > 0.0) {
            float inner_angle = light.angle;
            float outer_angle = inner_angle * 0.8; // 20% softer outer edge
            spot_factor = clamp((cos_theta - outer_angle) / (inner_angle - outer_angle), 0.0, 1.0);
        }

        // Diffuse
        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo_color;

        // Specular (Blinn-Phong)
        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_color = specular * light.color * specular_color;

        total_light += (diffuse_color + specular_color) * attenuation * spot_factor;
    }

    // Gamma correction (optional)
    total_light = pow(total_light, vec3(1.0/2.2));

    final_color = vec4(total_light, 1.0);
}