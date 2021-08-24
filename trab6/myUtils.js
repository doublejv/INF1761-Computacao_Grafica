class Sphere20{
    constructor( ){
        const X=0.525731112119133606;
        const Z=0.850650808352039932;
        this.vertices = [
            -X, 0.0, Z,   X, 0.0, Z,   -X, 0.0, -Z,   X, 0.0, -Z,
            0.0, Z, X,    0.0, Z, -X,   0.0, -Z, X,   0.0, -Z, -X,
            Z, X, 0.0,   -Z, X, 0.0,    Z, -X, 0.0,   -Z, -X, 0.0
        ]; 
        this.indices = [
            0,4,1,  0,9,4,  9,5,4,  4,5,8,  4,8,1, 
            8,10,1, 8,3,10, 5,3,8,  5,2,3,  2,7,3,
            7,10,3, 7,6,10, 7,11,6, 11,0,6, 0,1,6,
            6,1,10, 9,0,11, 9,11,2, 9,2,5,  7,2,11
        ]; 
        this.normals = this.vertices;
    }

    getVertices( ){
        return this.vertices;
    }

    getNormals(){
        return this.normals;
    }

    getIndices(){
        return this.indices;
    }

    getIndex(x,y,z){
        var nvertices = this.vertices.length/3;
        var TOL = 0.001;
        for(var k=0; k < nvertices; k++){
            if ((Math.abs(x-this.vertices[3*k+0])<TOL) && (Math.abs(y-this.vertices[3*k+1])<TOL) && (Math.abs(z-this.vertices[3*k+2])<TOL)){
                return k;
            }
        }
        this.vertices.push(x,y,z);
        return nvertices;
    }

    refine(){
        var ntriang = this.indices.length/3;
        for (var t=0; t < ntriang; t++){
            var i0 = this.indices[3*t+0];
            var j0 = this.indices[3*t+1];
            var k0 = this.indices[3*t+2];

            // criar mid side node i1
            var x = (this.vertices[3*j0+0]+this.vertices[3*k0+0])/2.0;
            var y = (this.vertices[3*j0+1]+this.vertices[3*k0+1])/2.0;
            var z = (this.vertices[3*j0+2]+this.vertices[3*k0+2])/2.0;
            var size = Math.sqrt(x*x+y*y+z*z);
            x /= size;
            y /= size;
            z /= size;
            var i1 = this.getIndex(x,y,z);

            // criar mid side node j1
            x = (this.vertices[3*i0+0]+this.vertices[3*k0+0])/2.0;
            y = (this.vertices[3*i0+1]+this.vertices[3*k0+1])/2.0;
            z = (this.vertices[3*i0+2]+this.vertices[3*k0+2])/2.0;
            size = Math.sqrt(x*x+y*y+z*z);
            x /= size;
            y /= size;
            z /= size;
            var j1 = this.getIndex(x,y,z);

            // criar mid side node k1
            x = (this.vertices[3*i0+0]+this.vertices[3*j0+0])/2.0;
            y = (this.vertices[3*i0+1]+this.vertices[3*j0+1])/2.0;
            z = (this.vertices[3*i0+2]+this.vertices[3*j0+2])/2.0;
            size = Math.sqrt(x*x+y*y+z*z);
            x /= size;
            y /= size;
            z /= size;
            var k1 = this.getIndex(x,y,z);

            // criar novos triângulos
            this.indices.push(i0,k1,j1);
            this.indices.push(j1,i1,k0);
            this.indices.push(k1,j0,i1);
            this.indices[3*t+0] = i1;
            this.indices[3*t+1] = j1;
            this.indices[3*t+2] = k1;
        }

    }
}

