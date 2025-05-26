import numpy as np
from PIL import Image



#~datko.pl
EPSILON = 0.0001


def reflect(vector, normal_vector):
    """Returns reflected vector."""
    n_dot_l = np.dot(vector, normal_vector)
    return vector - normal_vector * (2 * n_dot_l)


def normalize(vector):
    """Return normalized vector (length 1)."""
    return vector / np.sqrt((vector**2).sum())

class Ray:
    """Ray class."""
    def __init__(self, starting_point, direction):
        """Ray consist of starting point and direction vector."""
        self.starting_point = starting_point
        self.direction = direction


class Light:
    """Light class."""
    def __init__(self, position):
        """The constructor.

        :param np.array position: The position of light
        :param np.array ambient: Ambient light RGB
        :param np.array diffuse: Diffuse light RGB
        :param np.array specular: Specular light RGB
        """
        self.position = position
        self.ambient=np.array([0, 0, 0])
        self.diffuse=np.array([0, 1, 1])
        self.specular=np.array([1, 1, 0])


class SceneObject:
    """The base class for every scene object."""
    def __init__(
        self,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25
    ):
        """The constructor.

        :param np.array ambient: Ambient color RGB
        :param np.array diffuse: Diffuse color RGB
        :param np.array specular: Specular color RGB
        :param float shining: Shinging parameter (Phong model)
        """
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shining = shining

    def get_normal(self, cross_point):
        """Should return a normal vector in a given cross point.

        :param np.array cross_point
        """
        raise NotImplementedError

    def trace(self, ray):
        """Checks whenever ray intersect with an object.

        :param Ray ray: Ray to check intersection with.
        :return: tuple(cross_point, distance)
            cross_point is a point on a surface hit by the ray.
            distance is the distance from the starting point of the ray.
            If there is no intersection return (None, None).
        """
        raise NotImplementedError

    def get_color(self, cross_point, obs_vector, scene):
        """Returns a color of an object in a given point.

        :param np.array cross_point: a point on a surface
        :param np.array obs_vector: observation vector used in Phong model
        :param Scene scene: A scene object (for lights)
        """

        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))

        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:
            v_dot_r = 0

        if n_dot_l > 0:
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +
                (self.specular * light.specular * v_dot_r**self.shining) +
                (self.ambient * light.ambient)
            )

        return color


class Sphere(SceneObject):
    """An implementation of a sphere object."""
    def __init__(
        self,
        position,
        radius,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25
    ):
        """The constructor.

        :param np.array position: position of a center of a sphere
        :param float radius: a radius of a sphere
        """
        super(Sphere, self).__init__(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shining=shining
        )
        self.position = position
        self.radius = radius

    def get_normal(self, cross_point):
        return normalize(cross_point - self.position)

    def trace(self, ray):
        """Returns cross point and distance or None."""
        distance = ray.starting_point - self.position

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4*a*c

        if d < 0:
            return (None, None)

        sqrt_d = d**(0.5)
        denominator = 1 / (2 * a)

        if d > 0:
            r1 = (-b - sqrt_d) * denominator
            r2 = (-b + sqrt_d) * denominator
            if r1 < EPSILON:
                if r2 < EPSILON:
                    return (None, None)
                r1 = r2
        else:
            r1 = -b * denominator
            if r1 < EPSILON:
                return (None, None)

        cross_point = ray.starting_point + r1 * ray.direction
        return cross_point, r1


class Camera:
    """Implementation of a Camera object."""
    def __init__(
        self,
        position = np.array([0, 0, -3]),
        look_at = np.array([0, 0, 0]),
    ):
        """The constructor.

        :param np.array position: Position of the camera.
        :param np.array look_at: A point that the camera is looking at.
        """
        self.z_near = 1
        self.pixel_height = 500
        self.pixel_width = 700
        self.povy = 45
        look = normalize(look_at - position)
        self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
        self.position = position
        self.look_at = look_at
        self.direction = normalize(look_at - position)
        aspect = self.pixel_width / self.pixel_height
        povy = self.povy * np.pi / 180
        self.world_height = 2 * np.tan(povy/2) * self.z_near
        self.world_width = aspect * self.world_height

        center = self.position + self.direction * self.z_near
        width_vector = normalize(np.cross(self.up, self.direction))
        self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
        self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
        self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)

    def get_world_pixel(self, x, y):
        return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y


class Scene:
    """Container class for objects, light and camera."""
    def __init__(
        self,
        objects,
        light,
        camera
    ):
        self.objects = objects
        self.light = light
        self.camera = camera
        self.ambient = np.array([0.1, 0.1, 0.1])
        self.background = np.array([0, 0, 0])


class RayTracer:
    """RayTracer class."""
    def __init__(self, scene):
        """The constructor.

        :param Scene scene: scene to render
        """
        self.scene = scene

    def generate_image(self):
        """Generates image.

        :return np.array image: The computed image."
        """
        camera = self.scene.camera
        image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                world_pixel = camera.get_world_pixel(x, y)
                direction = normalize(world_pixel - camera.position)
                image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
        return image

    def _get_pixel_color(self, ray):
        """Gets a single color based on a ray.

        :return np.array: A hit object color or a background.
        """

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background

        return obj.get_color(cross_point, ray.direction, self.scene)

    def _get_closest_object(self, ray):
        """Finds the closes object to the ray.

        :return tuple(scene_object, distance, cross_point)
            scene_object SceneObject: hit object or None (if no hit)
            distance float: Distance to the the scene_object
            cross_point np.array: the interection point.
        """
        closest = None
        min_distance = np.inf
        min_cross_point = None
        for obj in self.scene.objects:
            cross_point, distance = obj.trace(ray)
            if cross_point is not None and distance < min_distance:
                min_distance = distance
                closest = obj
                min_cross_point = cross_point

        return (closest, min_distance, min_cross_point)


#zadania

def run_example():
    scene = Scene(
        objects=[Sphere(position=np.array([0, 0, 0]), radius=1.5)],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 5]))
    )

    rt = RayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)

    # Zamiana na format uint8
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.show()


def reference():
    scene = Scene(
        objects=[
            Sphere(position=np.array([-1, 0, -1]), radius=1),
            Sphere(position=np.array([1, 0,-1]), radius=1)

        ]
        ,
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 5]))
    )

    rt = RayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save("reference.png")
    img.show()


#zad1
def zad1():
    scene = Scene(
        objects=[
            Sphere(position=np.array([0, 0, 0]), radius=1.5),
            #dodatkowa kula
            Sphere(position=np.array([3, 0,-4]), radius=1)

        ]
        ,
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 5]))
    )

    rt = RayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save("render1.png")
    img.show()



#zad2
class MyRayTracer(RayTracer):
    def _get_pixel_color(self, ray, depth=3):

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background
        if depth == 0:
            return obj.get_color(cross_point, ray.direction, self.scene)

        #nowy promień
        new_ray = Ray(cross_point, reflect(ray.direction, normalize(obj.get_normal(cross_point))))

        #rekurencyjne wywołanie _get_pixel_color()
        return (0.75 * obj.get_color(cross_point, ray.direction, self.scene)
                + 0.25 * self._get_pixel_color(new_ray, depth=depth - 1)
                )


def zad2():
    scene = Scene(
        objects=[
            Sphere(position=np.array([0, 0, 0]), radius=1),
            Sphere(position=np.array([3, 0,-3]), radius=1, diffuse=np.array([0.2,0.9,0])),


        ]
        ,
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 5]))
    )

    rt = MyRayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save("render2.png")
    img.show()


#zad3
class MySphere(Sphere):
    def get_color(self, cross_point, obs_vector, scene):
        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))

        #sprawdzanie, czy na drodze promienia znajduje się obiekt
        interference = None
        for obj in scene.objects:
            first, second = obj.trace(Ray(cross_point, light_vector))
            if first is not None:
                interference = first
                break
        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:
            v_dot_r = 0

        #zwrócenie koloru, jeśli kosinus kąta między normalną
        # a kierunkiem światła oraz brak przeszkód na drodze promienia
        if n_dot_l > 0 and interference is None:
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +
                (self.specular * light.specular * v_dot_r**self.shining) +
                (self.ambient * light.ambient)
            )

        return color



def zad3():
    scene = Scene(
        objects=[
            MySphere(position=np.array([-1, 0, -3]), radius=1),
            MySphere(position=np.array([1, 0,-1]), radius=1)

        ]
        ,
        light=Light(position=np.array([5, 2, 5])),
        camera=Camera(position=np.array([3, 0, 5]))
    )

    rt = MyRayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save("render3.png")
    img.show()


#zad4
class MySphere_2(Sphere):
    def __init__(
            self,
            position,
            radius,
            clarity=0.0,
            refraction=0.0,
            ambient=np.array([0, 0, 0]),
            diffuse=np.array([0.6, 0.7, 0.8]),
            specular=np.array([0.8, 0.8, 0.8]),
            shining=25
    ):
        super(MySphere_2, self).__init__(
            position=position,
            radius=radius,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shining=shining
        )
        self.clarity = clarity
        self.refraction = refraction

    def trace(self, ray):
        distance = ray.starting_point - self.position
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4*a*c

        if d < 0:
            return (None, None)

        sqrt_d = d**(0.5)
        denominator = 1 / (2 * a)
        #obliczanie zawsze dwóch punktów przecięcia
        r1 = (-b - sqrt_d) * denominator
        r2 = (-b + sqrt_d) * denominator

        valid = []
        if r1 > EPSILON:
            valid.append(r1)
        if r2 > EPSILON:
            valid.append(r2)

        if not valid:
            return (None, None)

        r = min(valid)
        cross_point = ray.starting_point + r * ray.direction
        return (cross_point, r)

    def trace_refraction(self, ray):
        distance = ray.starting_point - self.position
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4*a*c

        if d < 0:
            return None, None

        sqrt_d = np.sqrt(d)
        denominator = 1 / (2 * a)
        r1 = (-b - sqrt_d) * denominator
        r2 = (-b + sqrt_d) * denominator

        if r1 > r2:
            r1, r2 = r2, r1

        if r2 < EPSILON:
            return None, None

        return r1, r2

class MyRayTracer2(RayTracer):
    def _get_pixel_color(self, ray, depth=3):
        if depth <= 0:
            return np.array([0, 0, 0])

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background

        normal = obj.get_normal(cross_point)
        reflected_dir = reflect(ray.direction, normal)
        reflected_ray = Ray(cross_point + normal * 1e-5, reflected_dir)

        local_obj_color = obj.get_color(cross_point, ray.direction, self.scene)
        reflected_pixel_color = self._get_pixel_color(reflected_ray, depth=depth - 1)

        if obj.clarity > 0 and obj.refraction > 0:
            r1, r2 = obj.trace_refraction(ray)
            if r1 is None or r2 is None:
                return 0.85 * local_obj_color + 0.15 * reflected_pixel_color

            exit_point = ray.starting_point + r2 * ray.direction
            exit_normal = obj.get_normal(exit_point)

            # sprawdzenie czy promień wchodzi czy wychodiz
            cos_theta = np.dot(ray.direction, exit_normal)
            if cos_theta < 0:
                # wchodzi
                eta = 1.0 / obj.refraction
            else:
                # wychodzi
                eta = obj.refraction
                exit_normal = -exit_normal

            #wykorzystanie prawa snella
            cosi = -np.dot(exit_normal, normalize(ray.direction))
            k = 1 - eta**2 * (1 - cosi**2)

            if k < 0:
                return local_obj_color * 0.4 + reflected_pixel_color * 0.6

            refracted_dir = eta * normalize(ray.direction) + (eta * cosi - k**(0.5)) * exit_normal
            refract_start = exit_point + exit_normal * 1e-5
            refracted_ray = Ray(refract_start, normalize(refracted_dir))
            refracted_color = self._get_pixel_color(refracted_ray, depth - 1)

            return (
                (1 - obj.clarity) * local_obj_color +
                0.3 * reflected_pixel_color +
                obj.clarity * refracted_color
            )
        else:
            return 0.85 * local_obj_color + 0.15 * reflected_pixel_color

def zad4():
    scene = Scene(
        objects=[
            MySphere_2(position=np.array([0, 0, -2]), radius=2, clarity=0.0, refraction=1.0),
            MySphere_2(position=np.array([1, 0, 2]), radius=1, diffuse=np.array([0.9, 0.7, 0.3]), clarity=0.7, refraction=1.02)
        ],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 8]))
    )

    rt = MyRayTracer2(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save("render4.png")
    img.show()


class Triangle(SceneObject):
    def __init__(self, a, b, c, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.normal = self._calculate_normal()

    def _calculate_normal(self):
        v1 = self.b - self.a
        v2 = self.c - self.a
        return normalize(np.cross(v1, v2))

    def get_normal(self, point):
        return self.normal

    def trace(self, ray):

        denominator = np.dot(self.normal, ray.direction)
        if abs(denominator) < EPSILON: return (None, None)

        t = np.dot(self.a - ray.starting_point, self.normal)/denominator
        if t < EPSILON: return (None, None)

        point = ray.starting_point + ray.direction*t

        #sprawdzanie czy punkt leży w trójkącie
        #analogicznie do lab4
        edge1 = self.b - self.a
        edge2 = self.c - self.b
        edge3 = self.a - self.c
        vp1 = point - self.a
        vp2 = point - self.b
        vp3 = point - self.c

        c1 = np.dot(np.cross(edge1, vp1), self.normal)
        c2 = np.dot(np.cross(edge2, vp2), self.normal)
        c3 = np.dot(np.cross(edge3, vp3), self.normal)

        if (c1 > 0 and c2 > 0 and c3 > 0) or (c1 < 0 and c2 < 0 and c3 < 0):
            return (point, t)
        return (None, None)

def zad5():
    scene = Scene(
        objects=[
            MySphere_2(
                position=np.array([0, 0.5, -2]),
                radius=2,
                diffuse=np.array([0.3, 0.7, 0.9])
            ),
            Triangle(
                a=np.array([0, -1, 2]),
                b=np.array([2, -1, 2]),
                c=np.array([1, 1, 2]),
                diffuse=np.array([0.5, 0.3, 0.7])
            )
        ],
        light=Light(position=np.array([3, 5, 5])),
        camera=Camera(
            position=np.array([0, 0, 8]),
            look_at=np.array([0, 0, 0])
        )
    )

    rt = MyRayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    img =Image.fromarray((image*255).astype(np.uint8))
    img.save("render5.png")
    img.show()


