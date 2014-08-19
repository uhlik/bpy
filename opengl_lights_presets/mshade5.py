import bpy
l0 = bpy.context.user_preferences.system.solid_lights[0]
l1 = bpy.context.user_preferences.system.solid_lights[1]
l2 = bpy.context.user_preferences.system.solid_lights[2]

l0.use = True
l0.diffuse_color = (0.4753793776035309, 0.4753793776035309, 0.4753793776035309)
l0.specular_color = (0.5, 0.5, 0.5)
l0.direction = (-0.012820512987673283, 0.44871795177459717, 0.8935815095901489)
l1.use = True
l1.diffuse_color = (0.7233469486236572, 0.7233469486236572, 0.7233469486236572)
l1.specular_color = (0.0, 0.0, 0.0)
l1.direction = (0.0, 0.9230769276618958, 0.38461539149284363)
l2.use = True
l2.diffuse_color = (0.1267620027065277, 0.12150080502033234, 0.11475051194429398)
l2.specular_color = (0.0, 0.0, 0.0)
l2.direction = (0.012177534401416779, -0.9742027521133423, -0.22534571588039398)
