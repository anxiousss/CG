#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position; float _pad0;
    vec3 ambient_light_intensity; float _pad1;
    vec3 sun_light_direction; float _pad2;
    vec3 sun_light_color; float _pad3;
    uint point_light_count;
    uint spot_light_count;
    float _pad4; float _pad5;
};

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color; float _pad6;
    vec3 specular_color; float _pad7;
    float shininess;  float _pad8;  float _pad9;  float _pad10;
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

layout(binding = 3, std430) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    // Ambient lighting
    vec3 total_light = ambient_light_intensity * albedo_color;

    // Sun lighting (directional light)
    vec3 sun_light_dir = normalize(-sun_light_direction);
    float sun_diffuse = max(dot(normal, sun_light_dir), 0.0);
    vec3 sun_half_dir = normalize(sun_light_dir + view_dir);
    float sun_specular = pow(max(dot(normal, sun_half_dir), 0.0), shininess);

    total_light += (sun_diffuse * albedo_color + sun_specular * specular_color) * sun_light_color;

    // Point lights with inverse square law attenuation
    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);

        // Skip if beyond radius
        if (distance > light.radius) continue;

        // Inverse square law attenuation with minimum distance to prevent singularity
        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);

        // Additional smooth falloff at radius boundary
        float radius_attenuation = 1.0 - smoothstep(light.radius * 0.7, light.radius, distance);
        attenuation *= radius_attenuation;

        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo_color;

        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_result = specular * light.color * specular_color;

        total_light += (diffuse_color + specular_result) * attenuation;
    }

    // Spot lights with inverse square law attenuation
    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);

        if (distance > light.radius) continue;

        vec3 spot_dir = normalize(-light.direction);
        float cos_theta = dot(light_dir, spot_dir);

        // Skip if outside spotlight cone
        if (cos_theta < light.angle) continue;

        // Inverse square law attenuation
        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);

        // Smooth angular falloff
        float outer_angle = light.angle * 0.8;
        float spot_factor = clamp((cos_theta - outer_angle) / (light.angle - outer_angle), 0.0, 1.0);

        // Smooth radius falloff
        float radius_attenuation = 1.0 - smoothstep(light.radius * 0.7, light.radius, distance);
        attenuation *= spot_factor * radius_attenuation;

        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo_color;

        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_result = specular * light.color * specular_color;

        total_light += (diffuse_color + specular_result) * attenuation;
    }

    // Tone mapping and gamma correction
    // total_light = total_light / (total_light + vec3(1.0));
    // total_light = pow(total_light, vec3(1.0/2.2));

    final_color = vec4(total_light, 1.0);
}