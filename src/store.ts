export interface Robot {
    teamname: string;
    drivespeed: number;
    uuid: string;
    user_uuid: string;
    update_at: Date;
    image_name: string;
    wheelsize: number;
    intaketype: string;
    intakespeed: number;
    tierhang: string;
    robotsize: number[];
}

export class RobotStore {
    public robots: Robot[];
    static robots: any;

    constructor() {
        this.robots = [];
    }

    // Add a new post to the store
    addPost(robt: Robot): void {
        this.robots.push(robt);
    }

    // Get all posts from the store
    getRobots(): Robot[] {
        return this.robots;
    }

    // Get a post by UUID
    getRobotByUUID(uuid: string): Robot | undefined {
        return this.robots.find(robt => robt.uuid === uuid);
    }
}

// Example usage:
// const postStore = new RobotStore();

// const newPost: Post = {
//     title: "Sample Post",
//     phoneNumber: 1234567890,
//     uuid: 12345678901234567890n // Example bigint value
// };

// postStore.addPost(newPost);

// console.log(postStore.getPosts());

// const foundPost = postStore.getPostByUUID(12345678901234567890n);
// console.log(foundPost);
